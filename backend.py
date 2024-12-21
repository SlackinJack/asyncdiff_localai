#!/usr/bin/env python3
import argparse
import backend_pb2
import backend_pb2_grpc
import base64
import grpc
import json
import os
import pickle
import requests
import signal
import subprocess
import sys
import time
import torch
from concurrent import futures
from pathlib import Path
from PIL import Image


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
COMPEL = os.environ.get("COMPEL", "0") == "1"
WARMUP_STEPS = 0
HOST_INIT_TIMEOUT = 600


# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


process = None


def kill_process():
    global process
    if process is not None:
        process.terminate()
        time.sleep(15)
        process = None


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    last_height = None
    last_width = None
    last_model_path = None
    last_cfg_scale = 7
    # last_scheduler = None

    is_loaded = False
    needs_reload = False


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        if request.CFGScale != 0 and self.last_cfg_scale != request.CFGScale:
            self.last_cfg_scale = request.CFGScale

        if request.Model != self.last_model_path or (
            len(request.ModelFile) > 0 and (
                os.path.exists(request.ModelFile) and (
                    request.ModelFile != self.last_model_path
                )
            )
        ):
            self.needs_reload = True

        self.last_model_path = request.Model
        if request.ModelFile != "":
            if os.path.exists(request.ModelFile):
                self.last_model_path = request.ModelFile

        # if request.SchedulerType != self.last_scheduler:
        #     self.needs_reload = True
        #     self.last_scheduler = request.SchedulerType

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.last_height or request.width != self.last_width:
            self.needs_reload = True

        if not self.is_loaded or self.needs_reload:
            kill_process()
            nproc_per_node = torch.cuda.device_count()
            self.last_height = request.height
            self.last_width = request.width

            # scheduler = self.last_scheduler
            # if scheduler is None or len(scheduler) == 0:
            #     scheduler = "ddim"

            pipeline_type = None
            model_name = self.last_model_path.lower()
            if "stable-video-diffusion-img2vid" in model_name:
                pipeline_type = "svd"
            else:
                kill_process()
                assert False, "Unknown model type"

            cmd = [
                'torchrun',
                f'--nproc_per_node={nproc_per_node}',
                'host.py',

                f'--model={self.last_model_path}',
                f'--pipeline_type={pipeline_type}',
                # f'--scheduler={scheduler}',
            ]

            cmd = [arg for arg in cmd if arg]
            global process
            process = subprocess.Popen(cmd)
            initialize_url = "http://localhost:6000/initialize"
            time_elapsed = 0
            while True:
                try:
                    response = requests.get(initialize_url)
                    if response.status_code == 200 and response.json().get("status") == "initialized":
                        self.is_loaded = True
                        self.needs_reload = False
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed > HOST_INIT_TIMEOUT:
                    kill_process()
                    return backend_pb2.Result(message=f"Failed to launch host within {HOST_INIT_TIMEOUT} seconds", success=False)

        if self.is_loaded:
            url = 'http://localhost:6000/generate'
            data = {
                "image": request.src,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.step,
                "seed": request.seed,
                "decode_chunk_size": 2, #8 default, lower to use less vram
                "num_frames": 25, #25 default
                "output_path": request.dst,
            }

            response = requests.post(url, json=data)
            response_data = response.json()
            output_path = response_data.get("output_path", "")

            if os.path.isfile(output_path):
                # TODO: get gif or mp4 or both
                return backend_pb2.Result(message="Media generated", success=True)
            return backend_pb2.Result(message="No media generated", success=False)
        else:
            return backend_pb2.Result(message="Host is not loaded", success=False)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        kill_process()
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        kill_process()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)

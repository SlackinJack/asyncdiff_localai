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
COMPEL              = os.environ.get("COMPEL", "0") == "1"
CHUNK_SIZE          = int(os.environ.get("CHUNK_SIZE", 8))
MOTION_BUCKET_ID    = int(os.environ.get("MOTION_BUCKET_ID", 180))
NOISE_AUG_STRENGTH  = float(os.environ.get("NOISE_AUG_STRENGTH", 0.01))
FRAMES              = int(os.environ.get("FRAMES", 25))
TIME_SHIFT          = os.environ.get("TIME_SHIFT", "0") == "1"
SCALE_PERCENTAGE    = int(os.environ.get("SCALE_PERCENTAGE", 80))
CONTROLNET_SCALE    = float(os.environ.get("CONTROLNET_SCALE", 1.0))
# WARMUP_STEPS        = 0
HOST_INIT_TIMEOUT   = 600


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
    last_pipeline_type = None
    last_scheduler = None
    variant = "fp32"
    loras = []
    last_controlnet = None
    last_clip_skip = 0
    is_low_vram = False

    is_loaded = False
    needs_reload = False


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        assert request.PipelineType in ["sd1", "sd2", "sd3", "sdup", "sdxl", "svd", "ad"], "Unsupported pipeline type"
        if request.PipelineType != self.last_pipeline_type: 
            self.last_pipeline_type = request.PipelineType
            self.needs_reload = True

        if request.CFGScale != 0 and self.last_cfg_scale != request.CFGScale:
            self.last_cfg_scale = request.CFGScale

        if request.Model != self.last_model_path or (
            len(request.ModelFile) > 0 and os.path.exists(request.ModelFile) and request.ModelFile != self.last_model_path
        ):
            self.needs_reload = True

        self.last_model_path = request.Model
        if request.ModelFile != "":
            if os.path.exists(request.ModelFile):
                self.last_model_path = request.ModelFile

        if request.F16Memory and self.variant != "fp16":
            self.needs_reload = True
            self.variant = "fp16"
        elif not request.F16Memory and self.variant != "fp32":
            self.needs_reload = True
            self.variant = "fp32"

        if request.SchedulerType != self.last_scheduler:
            self.needs_reload = True
            self.last_scheduler = request.SchedulerType

        if request.LoraAdapters:
            if len(self.loras) == 0 and len(request.LoraAdapters) == 0:
                pass
            elif len(self.loras) != len(request.LoraAdapters):
                self.needs_reload = True
            else:
                a = {}
                for adapter in self.loras:
                    for k, v in adapter.items():
                        a[k] = v
                for k, v in a.items():
                    if k not in request.LoraAdapters.keys():
                        self.needs_reload = True
                        break
                if not self.needs_reload:
                    for adapter in request.LoraAdapters.keys():
                        if adapter not in a.keys():
                            self.needs_reload = True
                            break
            if self.needs_reload:
                self.loras = []
                if len(request.LoraAdapters) > 0:
                    i = 0
                    for adapter in request.LoraAdapters:
                        self.loras.append({ "lora": adapter, "weight": request.LoraScales[i] })
                        i += 1

        if len(request.ControlNet) > 0 and request.ControlNet != self.last_controlnet:
            self.needs_reload = True
            self.last_controlnet = request.ControlNet

        if request.CLIPSkip != self.last_clip_skip:
            self.last_clip_skip = request.CLIPSkip

        if self.is_low_vram != request.LowVRAM:
            self.needs_reload = True
            self.is_low_vram = request.LowVRAM

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.last_height or request.width != self.last_width:
            self.needs_reload = True

        if not self.is_loaded or self.needs_reload:
            kill_process()
            nproc_per_node = torch.cuda.device_count()
            self.last_height = request.height
            self.last_width = request.width

            assert nproc_per_node > 1, "This backend requires at least 2 GPUs."
            assert nproc_per_node < 6, "This backend does not support more than 5 GPUs. You can set a limit using CUDA_VISIBLE_DEVICES."
            match nproc_per_node:
                case 2:
                    model_n = 2
                    stride = 1
                case 3:
                    model_n = 2
                    stride = 2
                case 4:
                    model_n = 3
                    stride = 2
                case 5:
                    model_n = 4
                    stride = 2
                case _:
                    return

            cmd = [
                'torchrun',
                f'--nproc_per_node={nproc_per_node}',
                'host.py',

                '--host_mode=localai',
                f'--model={self.last_model_path}',
                f'--pipeline_type={self.last_pipeline_type}',
                f'--model_n={model_n}',
                f'--stride={stride}',
                f'--time_shift={TIME_SHIFT}',
                f'--variant={self.variant}',
            ]

            if COMPEL:
                cmd.append('--compel')

            if len(self.loras) > 0:
                cmd.append(f'--lora={json.dumps(self.loras)}')

            if self.last_controlnet is not None:
                cmd.append(f'--controlnet={self.last_controlnet}')
                cmd.append(f'--controlnet_scale={CONTROLNET_SCALE}')

            if self.last_scheduler is not None and len(self.last_scheduler) > 0 and self.last_pipeline_type in ['sd1', 'sd2', 'sd3', 'sdup', 'sdxl']:
                cmd.append(f'--scheduler={self.last_scheduler}')

            if self.is_low_vram:
                # cmd.append('--enable_model_cpu_offload')          # breaks parallelism
                # cmd.append('--enable_sequential_cpu_offload')     # crash
                cmd.append('--enable_tiling')
                cmd.append('--enable_slicing')
                cmd.append('--xformers_efficient')
                cmd.append('--scale_input')
                cmd.append(f'--scale_percentage={SCALE_PERCENTAGE}')

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

            if self.last_pipeline_type in ['sdup', 'svd']:
                if self.is_low_vram and CHUNK_SIZE >= 2:
                    decode_chunk_size = 2
                else:
                    decode_chunk_size = CHUNK_SIZE
                data = {
                    "image": request.src,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.step,
                    "seed": request.seed,
                    "decode_chunk_size": decode_chunk_size,
                    "num_frames": FRAMES,
                    "motion_bucket_id": MOTION_BUCKET_ID,
                    "noise_aug_strength": NOISE_AUG_STRENGTH,
                    "output_path": request.dst,
                }
                if request.positive_prompt and len(request.positive_prompt) > 0:
                    data["positive_prompt"] = request.positive_prompt
                if request.negative_prompt and len(request.negative_prompt) > 0:
                    data["negative_prompt"] = request.negative_prompt
            else:
                data = {
                    "positive_prompt": request.positive_prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.step,
                    "seed": request.seed,
                    "clip_skip": self.last_clip_skip,
                    "output_path": request.dst,
                }

            response = requests.post(url, json=data)
            response_data = response.json()
            output_path = response_data.get("output_path", "")

            if os.path.isfile(output_path):
                # TODO: return gif or mp4, fix file extension
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

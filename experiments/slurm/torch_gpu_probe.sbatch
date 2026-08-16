#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time=00:10:00

#SBATCH -J qgru-torch

#SBATCH -o /private/home/vtejomur/compression/torch-gpu-%j.out

set -euo pipefail

echo "NODE=$(hostname)"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python3 - <<'INNER'

import sys

print("Python:", sys.version)

try:

    import torch

    print("Torch:", torch.__version__)

    print("Torch CUDA build:", torch.version.cuda)

    print("CUDA available:", torch.cuda.is_available())

    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))

        print("Capability:", torch.cuda.get_device_capability(0))

        x = torch.randn(4096, 4096, device="cuda")

        y = torch.randn(4096, 4096, device="cuda")

        z = x @ y

        torch.cuda.synchronize()

        print("CUDA_MATMUL_OK:", float(z[0,0]))

except Exception as e:

    print("TORCH_PROBE_FAILED:", repr(e))

INNER


# DARWIN Cluster Setup

## Key Commands
- Request GPU node (partition changes depending on which gpu you're using): 'srun --gpus=1 --partition=gpu-mi50 --pty bash'
- Load ROCm or CUDA: 'vpkg_require amd-rocm' or 'vpkg_require cuda'
- Activate environment: 'source /opt/shared/miniforge/...'

## GPU Details
- Partition: gpu-mi50
- GPU: AMD MI50
- VRAM: 31.98GB
- Framework: ROCm (not CUDA)
- Max job time: 7 days

## Common Issues

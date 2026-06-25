import torch
print(f"PyTorch {torch.__version__} | built against CUDA {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()} | cuDNN: {torch.backends.cudnn.version()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory/1024**3:.1f} GB | Compute capability: sm_{props.major}{props.minor}")
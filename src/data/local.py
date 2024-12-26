import torch

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())

# Get GPU device name
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Check CUDA version
print("CUDA Version:", torch.version.cuda)
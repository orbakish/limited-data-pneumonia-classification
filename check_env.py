"""Environment check script for the pneumonia classification project.

This script verifies that the required Python packages are installed and
that the environment is correctly set up for both local development and
Colab training.

It verifies that the installed stack can load and run the three project architectures.
Run the same way in both environments (local and Colab).

python check_env.py     # local venv
!python check_env.py     # Colab

Reports library versions, the active compute device, and confirms each model loads pretrained weights and completes a forward pass successfully.    
"""

import importlib
import platform

import torch
from torchvision import models

# List of required Python packages and model architectures to check.
PACKAGES = ["torch", "torchvision", "numpy", "pandas", "matplotlib", "sklearn", "PIL", "tqdm"]
# Dictionary mapping model names to their corresponding torchvision model constructors.
MODELS = {
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnet18": models.resnet18,
    "efficientnet_b0": models.efficientnet_b0,
}

def report_versions():
    """Print the installed version of each required package."""
    print("--- versions ---")
    print(f" {'python':12s} {platform.python_version()}")
    # Check versions of required Python packages.
    for name in PACKAGES:
        # Attempt to import the package and retrieve its version.
        try:
            module = importlib.import_module(name)
            # Retrieve the version attribute of the module, defaulting to "unknown" if not found.
            version = getattr(module, "__version__", "unknown")
            print(f" {name:12s} {version}")
        # Handle the case where the package is not installed.
        except ImportError:
            print(f" {name:12s} NOT INSTALLED")

def get_device():
    """Return the torch device to use, plus a readable description of it."""
    # Check if a CUDA-enabled GPU is available and return the appropriate device and its name.
    if torch.cuda.is_available():
        # Use the first available CUDA device.
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    # Fallback to CPU if no CUDA-enabled GPU is available.
    return torch.device("cpu"), "CPU"

def check_model(name, builder, device):
    """Load a pretrained model and run one forward pass to verify it works."""
    # Build the model with pretrained weights and move it to the specified device.
    model = builder(weights="DEFAULT") 
    model.eval().to(device)
    # Count the total number of parameters in the model.
    n_params = sum(p.numel() for p in model.parameters())
    # Create a dummy input tensor with the appropriate shape for the model.
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        # Perform a forward pass with the dummy input to verify the model works.
        output = model(dummy)
    # Return the output shape and the number of parameters.
    return tuple(output.shape), n_params

def main():
    # Report the versions of Python and required packages.
    report_versions()

    # Get the device to use for model inference and print its description.
    device, device_name = get_device()
    print(f"\n--- device ---")
    print(f" {device.type}: {device_name}")

    # Check each model by running a forward pass with a dummy input.
    print(f"\n--- models ---")
    for name, builder in MODELS.items():
        shape, n_params = check_model(name, builder, device)
        print(f" {name:20s} output={shape} params={n_params:,}")

    print("\n Environment OK.")

if __name__ == "__main__":
    main()
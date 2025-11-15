#!/bin/bash

echo "========================================"
echo "MNIST Diffusion - Mojo"
echo "========================================"
echo ""

# Check if Mojo is installed
if ! command -v mojo &> /dev/null; then
    echo "Error: Mojo not found. Please install Mojo first."
    echo "Visit: https://www.modular.com/mojo"
    exit 1
fi



echo "Building and running Mojo diffusion model..."
echo ""

# Run the Mojo program
mojo run mnist_diffusion_mojo.mojo

echo ""
echo "========================================"
echo "Done!"
echo "========================================"

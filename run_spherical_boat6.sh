#!/bin/bash

# Script to run spherical projection on boat6 dataset
echo "Running spherical projection on boat6 dataset..."

cd "$(dirname "$0")"
python src/python/manual/manual_stitcher.py --dataset "boat6" --projection "spherical"

echo "Spherical projection on boat6 completed!" 
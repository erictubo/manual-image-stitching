#!/bin/bash

# Script to run automatic stitching on boat6 dataset
echo "Running automatic stitching on boat6 dataset..."

cd "$(dirname "$0")"
python src/python/automatic/auto_stitcher.py --dataset "boat6"

echo "Automatic stitching completed!" 
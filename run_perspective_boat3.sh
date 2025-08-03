#!/bin/bash

# Script to run perspective projection on boat3 dataset
echo "Running perspective projection on boat3 dataset..."

cd "$(dirname "$0")"
python src/python/manual/manual_stitcher.py --dataset "boat3" --projection "perspective"

echo "Perspective projection on boat3 completed!" 
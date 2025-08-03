# Step-by-step Image Stitching

This repository contains image stitching implementations in Python with both automatic and manual (from-scratch) approaches.

## Quick Start

```bash
# Automatic stitching (OpenCV built-in)
./run_automatic.sh

# Manual stitching - from-scratch implementation
./run_perspective_boat3.sh
./run_spherical_boat6.sh
```

## Repository Structure

```
Image Stitching/
├── images/
│   ├── boat3/          # 3-image boat dataset
│   └── boat6/          # 6-image boat dataset
├── src/
│   └── python/
│       ├── automatic/
│       │   ├── auto_stitcher.py
│       │   └── output/
│       └── manual/
│           ├── manual_stitcher.py
│           ├── output/
│           └── steps/
├── run_automatic.sh
├── run_perspective_boat3.sh
└── run_spherical_boat6.sh
```

## Stitching Methods

### Automatic Stitching
Uses OpenCV's built-in `cv2.Stitcher` - a black-box solution that automatically:
- Detects features and matches them
- Estimates homography transformations  
- Warps and blends images

### Manual Stitching (From-Scratch)
A complete from-scratch implementation that gives you full control over each step:
1. **Feature Detection**: SIFT keypoint detection
2. **Feature Matching**: FLANN matching with Lowe's ratio test
3. **Homography Estimation**: RANSAC-based homography calculation
4. **Image Warping**: Perspective transformation
5. **Blending**: Smooth transition between overlapping regions

**Why "Manual"?** This implementation builds the stitching pipeline from the ground up, allowing you to understand and modify each step of the process.

## Projection Types

- **Perspective**: Standard flat panoramas
- **Spherical**: 360° panoramas with automatic focal length detection

## Usage

### Shell Scripts (Recommended)
```bash
# Automatic (OpenCV built-in)
./run_automatic.sh

# Manual from-scratch implementations
./run_perspective_boat3.sh
./run_spherical_boat6.sh
```

### Direct Python Execution
```bash
# Automatic
cd src/python/automatic && python auto_stitcher.py

# Manual from-scratch
cd src/python/manual && python manual_stitcher.py --dataset boat3 --projection perspective
cd src/python/manual && python manual_stitcher.py --dataset boat6 --projection spherical
```

### Command Line Arguments

#### Manual Stitcher (`manual_stitcher.py`)
- `--dataset`: Dataset name (e.g., `boat3`, `boat6`)
- `--projection`: Projection type (`perspective` or `spherical`)
- `--focal_length`: Focal length for spherical projection in mm (default: 25.0)
- `--reference_index`: Index of reference image (defaults to middle image)
- `--no-visualize`: Disable step-by-step visualization (enabled by default)

## Output

- **Automatic**: `src/python/automatic/output/panorama.jpg`
- **Manual**: `src/python/manual/output/<dataset>/`
- **Step-by-step visualization**: `src/python/manual/steps/<dataset>/`

## Dependencies

- OpenCV (cv2)
- NumPy
- Pathlib

## C++ Implementation

**Note**: C++ implementation is planned but not yet implemented. The current focus is on the Python implementation.
import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from manual_stitcher import ManualImageStitcher, StitchingConfig

def find_repo_root():
    """Find the repository root directory by looking for the images folder"""
    current_dir = Path(__file__).parent
    repo_root = current_dir
    
    # Walk up the directory tree looking for the images folder
    while repo_root.parent != repo_root:  # Stop at filesystem root
        if (repo_root / "images").exists():
            return repo_root
        repo_root = repo_root.parent
    
    # If not found, return current directory
    return current_dir

def get_default_image_dir(dataset="boat"):
    """Get the default image directory path for a specific dataset"""
    repo_root = find_repo_root()
    return str(repo_root / "images" / dataset)

def get_output_dir(dataset="boat", output_type="output"):
    """Get the output directory path relative to the script location"""
    script_dir = Path(__file__).parent
    return str(script_dir / output_type / dataset)

def test_manual_stitching(input_dir, output_dir, test_output_dir, max_images=3):
    """Test different stitching methods"""
    
    # Initialize stitcher WITHOUT visualization first
    config = StitchingConfig()
    stitcher = ManualImageStitcher(config, visualize=False)
    
    # Load images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.JPG")))
    image_files.sort()
    
    print(f"Looking for images in: {input_dir}")
    print(f"Found {len(image_files)} image files")
    
    images = []
    for i, file_path in enumerate(image_files[:max_images]):  # Use first N images for better panorama
        print(f"Attempting to load: {file_path}")
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (600, 400))
            images.append(img)
            print(f"✓ Loaded image {i+1}: {os.path.basename(file_path)} (shape: {img.shape})")
        else:
            print(f"✗ Failed to load image: {file_path}")
    
    if len(images) < 2:
        print("Need at least 2 images for stitching")
        return
    
    # Create output directory
    output_dir_path = get_output_dir(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Test 1: Sequential stitching (original method)
    print("\n=== Test 1: Sequential Stitching ===")
    result_sequential = stitcher.stitch_panorama_sequential(images)
    if result_sequential is not None:
        cv2.imwrite(os.path.join(output_dir_path, "sequential_panorama.jpg"), result_sequential)
        print("✓ Sequential stitching completed!")
    else:
        print("✗ Sequential stitching failed!")
    
    # Test 2: Reference-based stitching (middle image as reference)
    print("\n=== Test 2: Reference-based Stitching (middle image) ===")
    result_reference_middle = stitcher.stitch_panorama_reference(images, reference_index=1)
    if result_reference_middle is not None:
        cv2.imwrite(os.path.join(output_dir_path, "reference_middle_panorama.jpg"), result_reference_middle)
        print("✓ Reference-based stitching (middle) completed!")
    else:
        print("✗ Reference-based stitching (middle) failed!")
    
    # Test 3: Reference-based stitching (first image as reference)
    print("\n=== Test 3: Reference-based Stitching (first image) ===")
    result_reference_first = stitcher.stitch_panorama_reference(images, reference_index=0)
    if result_reference_first is not None:
        cv2.imwrite(os.path.join(output_dir_path, "reference_first_panorama.jpg"), result_reference_first)
        print("✓ Reference-based stitching (first) completed!")
    else:
        print("✗ Reference-based stitching (first) failed!")
    
    # Test 4: Adaptive stitching (tries reference first, falls back to sequential)
    print("\n=== Test 4: Adaptive Stitching ===")
    result_adaptive = stitcher.stitch_panorama(images, method="adaptive")
    if result_adaptive is not None:
        cv2.imwrite(os.path.join(output_dir_path, "adaptive_panorama.jpg"), result_adaptive)
        print("✓ Adaptive stitching completed!")
    else:
        print("✗ Adaptive stitching failed!")
    
    # Test with visualization
    print("\n=== Test 5: Visualization Test ===")
    test_output_dir_path = get_output_dir(test_output_dir)
    stitcher_with_viz = ManualImageStitcher(config, visualize=True, test_output_dir=test_output_dir_path)
    
    # Test reference-based stitching with visualization
    if len(images) >= 3:
        print("Creating panorama with 3 images using reference method...")
        result_with_viz = stitcher_with_viz.stitch_panorama_reference(images, reference_index=1)
    else:
        print("Creating panorama with 2 images using reference method...")
        result_with_viz = stitcher_with_viz.stitch_panorama_reference(images, reference_index=0)
    
    if result_with_viz is not None:
        print("✓ Stitching with visualization completed!")
        cv2.imwrite(os.path.join(output_dir_path, "manual_panorama_with_viz.jpg"), result_with_viz)
        print(f"Result saved as {os.path.join(output_dir_path, 'manual_panorama_with_viz.jpg')}")
        print("Note: Visualization windows were shown during the process.")
        
        # Show final result
        cv2.imshow("Final Panorama", result_with_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ Stitching with visualization failed!")
    
    print("\n=== Test Summary ===")
    print(f"All test results saved in {output_dir_path}/ directory:")
    print("- sequential_panorama.jpg")
    print("- reference_middle_panorama.jpg") 
    print("- reference_first_panorama.jpg")
    print("- adaptive_panorama.jpg")
    print("- manual_panorama_with_viz.jpg")

def test_image_order_independence(input_dir, output_dir, max_images=3):
    """Test that stitching works regardless of image order"""
    
    print("\n=== Testing Image Order Independence ===")
    
    # Initialize stitcher
    config = StitchingConfig()
    stitcher = ManualImageStitcher(config, visualize=False)
    
    # Load images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.JPG")))
    image_files.sort()
    
    images = []
    for file_path in image_files[:max_images]:
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (600, 400))
            images.append(img)
    
    if len(images) < 3:
        print("Need at least 3 images for order independence test")
        return
    
    # Test different image orders
    orders = [
        ("Original", images),
        ("Reversed", list(reversed(images))),
        ("Middle first", [images[1], images[0], images[2]]),
        ("Last first", [images[2], images[0], images[1]])
    ]
    
    output_dir_path = get_output_dir(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    
    for order_name, ordered_images in orders:
        print(f"\nTesting {order_name} order...")
        
        # Use reference-based stitching with middle image as reference
        result = stitcher.stitch_panorama_reference(ordered_images, reference_index=1)
        
        if result is not None:
            filename = os.path.join(output_dir_path, f"order_test_{order_name.lower().replace(' ', '_')}.jpg")
            cv2.imwrite(filename, result)
            print(f"✓ {order_name} order stitching successful!")
        else:
            print(f"✗ {order_name} order stitching failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image stitching.")
    parser.add_argument("--dataset", type=str, default="boat", help="Dataset name (default: boat)")
    parser.add_argument("--max_images", type=int, default=3, help="Maximum number of images to use for stitching.")
    
    args = parser.parse_args()
    
    # Get paths based on dataset
    input_dir = get_default_image_dir(args.dataset)
    output_dir = get_output_dir(args.dataset, "output")
    test_output_dir = get_output_dir(args.dataset, "steps")
    
    test_manual_stitching(input_dir, output_dir, test_output_dir, args.max_images)
    test_image_order_independence(input_dir, output_dir, args.max_images) 
import cv2
import numpy as np
import os
import glob
from manual_stitcher import ManualImageStitcher, StitchingConfig

def test_manual_stitching():
    """Simple test of manual stitching"""
    
    # Initialize stitcher WITHOUT visualization first
    config = StitchingConfig()
    stitcher = ManualImageStitcher(config, visualize=False)
    
    # Load images - use robust path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(os.path.dirname(script_dir), "images")
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_files.sort()
    
    print(f"Looking for images in: {image_dir}")
    print(f"Found {len(image_files)} image files")
    
    images = []
    for i, file_path in enumerate(image_files[:3]):  # Use first 3 images for better panorama
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
    
    # Test with visualization
    print("\nTesting with visualization...")
    stitcher.visualize = True
    
    # Stitch multiple images for a better panorama
    if len(images) >= 3:
        print("Creating panorama with 3 images...")
        result_with_viz = stitcher.stitch_multiple(images)
    else:
        print("Creating panorama with 2 images...")
        result_with_viz = stitcher.stitch_pair(images[0], images[1])
    
    if result_with_viz is not None:
        print("Stitching with visualization completed!")
        cv2.imwrite("output/manual_panorama_with_viz.jpg", result_with_viz)
        print("Result saved as output/manual_panorama_with_viz.jpg")
        print("Note: Visualization windows were shown during the process.")
        
        # Show final result
        cv2.imshow("Final Panorama", result_with_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching with visualization failed!")

if __name__ == "__main__":
    test_manual_stitching() 
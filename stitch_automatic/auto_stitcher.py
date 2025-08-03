import cv2
import numpy as np
import os
import glob
from pathlib import Path

class ImageStitcher:
    def __init__(self):
        # Initialize the OpenCV stitcher
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        
    def load_images(self, image_dir, max_width=3000, max_height=3000):
        """Load images from directory and sort them"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        # Sort files to ensure consistent order
        image_files.sort()
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        print(f"Found {len(image_files)} images: {[os.path.basename(f) for f in image_files]}")
        
        # Load and resize images
        images = []
        for i, file_path in enumerate(image_files):
            img = cv2.imread(file_path)
            if img is not None:
                # Resize image if it's too large
                height, width = img.shape[:2]
                if width > max_width or height > max_height:
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                    print(f"Image {i+1}: Resized {os.path.basename(file_path)} from {width}x{height} to {new_width}x{new_height}")
                else:
                    print(f"Image {i+1}: Loaded {os.path.basename(file_path)} - Shape: {img.shape}")
                
                images.append(img)
            else:
                print(f"Warning: Could not load {file_path}")
        
        print(f"Total images loaded: {len(images)}")
        return images
    
    def stitch_images(self, images):
        """Stitch multiple images into a panorama"""
        if len(images) < 2:
            raise ValueError("At least 2 images are required for stitching")
        
        print(f"Starting to stitch {len(images)} images...")
        
        # Perform stitching
        status, panorama = self.stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print("Stitching completed successfully!")
            return panorama
        else:
            print(f"Stitching failed with status: {status}")
            return None
    
    def save_panorama(self, panorama, output_path):
        """Save the stitched panorama"""
        if panorama is not None:
            cv2.imwrite(output_path, panorama)
            print(f"Panorama saved to: {output_path}")
            print(f"Panorama dimensions: {panorama.shape}")
        else:
            print("No panorama to save")
    
    def display_panorama(self, panorama, window_name="Stitched Panorama"):
        """Display the panorama in a window"""
        if panorama is not None:
            # Resize if too large for display
            height, width = panorama.shape[:2]
            max_display_width = 1920
            max_display_height = 1080
            
            if width > max_display_width or height > max_display_height:
                scale = min(max_display_width / width, max_display_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_img = cv2.resize(panorama, (new_width, new_height))
                print(f"Resized for display: {new_width}x{new_height}")
            else:
                display_img = panorama
            
            cv2.imshow(window_name, display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No panorama to display")

def main():
    # Initialize the stitcher
    stitcher = ImageStitcher()
    
    # Set paths
    image_dir = "../images"
    output_dir = "output"
    output_path = os.path.join(output_dir, "panorama.jpg")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load images
        print("Loading images...")
        images = stitcher.load_images(image_dir)
        
        if len(images) < 2:
            print("Error: Need at least 2 images for stitching")
            return
        
        # Stitch images
        print("Stitching images...")
        panorama = stitcher.stitch_images(images)
        
        if panorama is not None:
            # Save the panorama
            stitcher.save_panorama(panorama, output_path)
            
            # Display the panorama
            print("Press any key to close the display window...")
            stitcher.display_panorama(panorama)
            
        else:
            print("Failed to create panorama")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
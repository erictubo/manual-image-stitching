import cv2
import numpy as np
import os
import glob
from pathlib import Path

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

def get_default_image_dir():
    """Get the default image directory path"""
    repo_root = find_repo_root()
    return str(repo_root / "images" / "boat")

def get_output_dir(output_dir, repo_root=None):
    """Get the output directory path, making it relative to repo root if it's a relative path"""
    if repo_root is None:
        repo_root = find_repo_root()
    
    # If output_dir is a relative path, make it relative to repo root
    if not os.path.isabs(output_dir):
        return str(repo_root / output_dir)
    return output_dir

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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automatic Image Stitcher')
    parser.add_argument('--input_dir', type=str, default=get_default_image_dir(),
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output panorama')
    parser.add_argument('--output_name', type=str, default='panorama.jpg',
                       help='Name of output panorama file')
    parser.add_argument('--max_width', type=int, default=3000,
                       help='Maximum width for image resizing')
    parser.add_argument('--max_height', type=int, default=3000,
                       help='Maximum height for image resizing')
    parser.add_argument('--display', action='store_true',
                       help='Display the panorama after stitching')
    
    args = parser.parse_args()
    
    # Initialize the stitcher
    stitcher = ImageStitcher()
    
    # Set paths
    output_path = os.path.join(get_output_dir(args.output_dir), args.output_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(get_output_dir(args.output_dir), exist_ok=True)
    
    try:
        # Load images
        print("Loading images...")
        images = stitcher.load_images(args.input_dir, args.max_width, args.max_height)
        
        if len(images) < 2:
            print("Error: Need at least 2 images for stitching")
            return
        
        # Stitch images
        print("Stitching images...")
        panorama = stitcher.stitch_images(images)
        
        if panorama is not None:
            # Save the panorama
            stitcher.save_panorama(panorama, output_path)
            
            # Display the panorama if requested
            if args.display:
                print("Press any key to close the display window...")
                stitcher.display_panorama(panorama)
            
        else:
            print("Failed to create panorama")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
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

def get_default_image_dir(dataset):
    """Get the default image directory path for a specific dataset"""
    repo_root = find_repo_root()
    return str(repo_root / "images" / dataset)

def get_output_dir(dataset, output_type="output"):
    """Get the output directory path relative to the script location"""
    script_dir = Path(__file__).parent
    return str(script_dir / output_type / dataset)

class StitchingConfig:
    """Configuration class for stitching parameters"""
    def __init__(self):
        self.feature_detector = 'SIFT'  # or 'SURF', 'ORB'
        self.matcher_type = 'FLANN'     # or 'BF'
        self.match_ratio = 0.75
        self.ransac_threshold = 5.0
        self.blend_width = 50
        self.min_matches = 10
        self.projection_type = 'perspective'  # 'perspective' or 'spherical'
        self.focal_length = 25.0  # Real focal length in mm

class ImageStitcher:
    """Image stitching with intelligent reference-based approach"""
    
    def __init__(self, config=None, visualize=False, test_output_dir="test_output"):
        """
        Initialize the stitcher
        
        Args:
            config: StitchingConfig object with parameters
            visualize: if True, shows intermediate steps
            test_output_dir: directory for saving visualization steps
        """
        self.config = config or StitchingConfig()
        self.visualize = visualize
        self.step_counter = 0  # Track visualization steps
        self.test_output_dir = test_output_dir
        
        # Create test output directory once if visualization is enabled
        if self.visualize:
            os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Initialize feature detector
        if self.config.feature_detector == 'SIFT':
            self.feature_detector = cv2.SIFT_create()
        elif self.config.feature_detector == 'ORB':
            self.feature_detector = cv2.ORB_create()
        else:
            self.feature_detector = cv2.SIFT_create()
        
        # Initialize matcher
        if self.config.matcher_type == 'FLANN':
            # FLANN parameters for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher()
        
        self.match_ratio = self.config.match_ratio
        self.ransac_threshold = self.config.ransac_threshold
        self.blend_width = self.config.blend_width
        self.min_matches = self.config.min_matches
        self.projection_type = self.config.projection_type
        self.focal_length = self.config.focal_length
        # Remove arbitrary sphere dimensions - they'll be calculated automatically

    def detect_features(self, image):
        """Detect keypoints and extract descriptors from a single image"""
        if image is None:
            print("Error: Input image is None")
            return [], None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        # Handle case where no features are detected
        if keypoints is None:
            keypoints = []
        if descriptors is None:
            descriptors = np.array([])
        
        print(f"Detected {len(keypoints)} keypoints")
        
        # Visualize if enabled
        if self.visualize:
            self.step_counter += 1
            img_with_keypoints = self.visualize_keypoints(image, keypoints)
            
            # Save visualization
            cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_keypoints.jpg", img_with_keypoints)
            
            cv2.imshow(f"Keypoints ({len(keypoints)} found)", img_with_keypoints)
            cv2.waitKey(500)  # Wait 0.5 seconds for better flow
            cv2.destroyAllWindows()
        
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2, image1=None, image2=None, keypoints1=None, keypoints2=None):
        """Match features between two sets of descriptors"""
        # Check if descriptors are valid
        if descriptors1 is None or descriptors2 is None:
            print("Error: Invalid descriptors provided")
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            print("Error: Empty descriptors provided")
            return []
        
        try:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            # Filter out None matches (can happen with some matchers)
            valid_matches = []
            for match_group in matches:
                if match_group is not None and len(match_group) == 2:
                    valid_matches.append(match_group)
            
            print(f"Found {len(valid_matches)} valid match groups")
            
            # Store data for visualization in filter_matches
            if self.visualize and image1 is not None and image2 is not None:
                self._last_matches_data = (image1, image2, keypoints1, keypoints2)
            
            return valid_matches
            
        except Exception as e:
            print(f"Error during feature matching: {e}")
            return []

    def filter_matches(self, matches):
        """Filter matches using Lowe's ratio test"""
        if not matches:
            print("No matches to filter")
            return []
        
        good_matches = []
        for match_group in matches:
            if len(match_group) == 2:
                m, n = match_group
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        print(f"Filtered to {len(good_matches)} good matches")
        
        # Visualize if enabled
        if self.visualize and hasattr(self, '_last_matches_data'):
            self.step_counter += 1
            image1, image2, keypoints1, keypoints2 = self._last_matches_data
            img_matches = self.visualize_matches(image1, image2, keypoints1, keypoints2, good_matches)
            
            # Save visualization
            cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_feature_matches.jpg", img_matches)
            
            cv2.imshow(f"Feature Matches ({len(good_matches)} good)", img_matches)
            cv2.waitKey(500)  # Wait 0.5 seconds for better flow
            cv2.destroyAllWindows()
        
        return good_matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        """Estimate homography matrix between two sets of keypoints"""
        if len(matches) < self.min_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            return None, None
        
        # Extract matched keypoints
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            print("Failed to estimate homography")
            return None, None
        
        # Validate homography quality
        if mask is not None:
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(matches)
            print(f"Homography quality: {inliers}/{len(matches)} inliers ({inlier_ratio:.2f})")
            
            if inlier_ratio < 0.5:
                print("Warning: Low inlier ratio, homography may be unreliable")
        
        return H, mask

    def calculate_stitching_bounds(self, image1, image2, homography):
        """Calculate the bounds for the stitched panorama"""
        # Get image dimensions
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Transform corners of image2 using homography
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, homography)
        
        # Get corners of image1 (reference)
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        
        # Combine all corners to find bounding box
        all_corners = np.vstack([corners1, warped_corners2.reshape(-1, 2)])
        
        # Calculate bounds
        min_x = int(np.floor(all_corners[:, 0].min()))
        max_x = int(np.ceil(all_corners[:, 0].max()))
        min_y = int(np.floor(all_corners[:, 1].min()))
        max_y = int(np.ceil(all_corners[:, 1].max()))
        
        # Calculate output dimensions and offset
        output_width = max_x - min_x
        output_height = max_y - min_y
        offset_x = -min_x  # How much to shift image1
        offset_y = -min_y  # How much to shift image1
        
        # Ensure minimum size
        if output_width < max(w1, w2):
            output_width = max(w1, w2)
        if output_height < max(h1, h2):
            output_height = max(h1, h2)
        
        return output_width, output_height, offset_x, offset_y

    def warp_image(self, image, homography, output_size, offset_x=0, offset_y=0):
        """Warp an image using homography matrix with optional offset"""
        if image is None or homography is None:
            print("Error: Invalid input for warping")
            return None
        
        if output_size[0] <= 0 or output_size[1] <= 0:
            print("Error: Invalid output size")
            return None
        
        try:
            # If we have an offset, create a translation matrix and combine with homography
            if offset_x != 0 or offset_y != 0:
                # Create translation matrix
                translation_matrix = np.array([
                    [1, 0, offset_x],
                    [0, 1, offset_y],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Combine homography with translation
                combined_matrix = translation_matrix @ homography
                warped = cv2.warpPerspective(image, combined_matrix, output_size)
            else:
                warped = cv2.warpPerspective(image, homography, output_size)
            
            print(f"Warped image to size: {output_size[0]}x{output_size[1]}")
            
            # Visualize if enabled
            if self.visualize:
                self.step_counter += 1
                
                # Save visualization
                cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_warped_image.jpg", warped)
                
                cv2.imshow(f"Warped Image ({output_size[0]}x{output_size[1]})", warped)
                cv2.waitKey(500)  # Wait 0.5 seconds for better flow
                cv2.destroyAllWindows()
            
            return warped
            
        except Exception as e:
            print(f"Error during image warping: {e}")
            return None

    def create_panorama(self, image1, warped_image, offset_x, offset_y):
        """Create panorama by combining image1 and warped image with smooth transition"""
        # Get dimensions
        h1, w1 = image1.shape[:2]
        warped_h, warped_w = warped_image.shape[:2]
        
        # Create output canvas - start with the warped image
        panorama = warped_image.copy()
        
        # Calculate the region where image1 should be placed
        img1_start_x = offset_x
        img1_end_x = offset_x + w1
        img1_start_y = offset_y
        img1_end_y = offset_y + h1
        
        # Find the overlap region
        overlap_start = max(0, offset_x)
        overlap_end = min(warped_w, offset_x + w1)
        overlap_width = overlap_end - overlap_start
        
        if overlap_width > 0:
            # Create smooth transition in overlap area using vectorized operations
            
            # Calculate the valid region for image1 placement
            valid_start_x = max(0, img1_start_x)
            valid_end_x = min(warped_w, img1_end_x)
            valid_start_y = max(0, img1_start_y)
            valid_end_y = min(warped_h, img1_end_y)
            
            # Calculate corresponding regions in image1
            img1_x_start = valid_start_x - offset_x
            img1_x_end = valid_end_x - offset_x
            img1_y_start = valid_start_y - offset_y
            img1_y_end = valid_end_y - offset_y
            
            # Extract the valid regions
            panorama_region = panorama[valid_start_y:valid_end_y, valid_start_x:valid_end_x]
            img1_region = image1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
            
            # Create alpha blending mask for overlap area
            if overlap_start < valid_end_x and overlap_end > valid_start_x:
                # Calculate overlap region within the valid area
                overlap_in_valid_start = max(overlap_start, valid_start_x)
                overlap_in_valid_end = min(overlap_end, valid_end_x)
                
                # Create alpha mask (1.0 on left, 0.0 on right of overlap)
                alpha_mask = np.ones_like(panorama_region, dtype=np.float32)
                
                # Calculate alpha values for the overlap region
                if overlap_in_valid_end > overlap_in_valid_start:
                    overlap_width_in_region = overlap_in_valid_end - overlap_in_valid_start
                    for i in range(overlap_width_in_region):
                        alpha = 1.0 - (i / overlap_width_in_region)
                        col_idx = overlap_in_valid_start - valid_start_x + i
                        if 0 <= col_idx < alpha_mask.shape[1]:
                            alpha_mask[:, col_idx] = alpha
                
                # Improved blending logic to prevent dark areas
                # Check if both images have meaningful content
                warped_has_content = np.any(panorama_region > 5, axis=2, keepdims=True)  # Lower threshold
                img1_has_content = np.any(img1_region > 5, axis=2, keepdims=True)  # Check image1 too
                
                # Create blended result
                blended = (alpha_mask * img1_region + 
                          (1 - alpha_mask) * panorama_region).astype(np.uint8)
                
                # Use blended result where both images have content
                # Otherwise, use the image that has content
                both_have_content = warped_has_content & img1_has_content
                panorama_region = np.where(both_have_content, blended, 
                                         np.where(warped_has_content, panorama_region, img1_region))
            else:
                # No overlap in valid region, just use image1
                panorama_region = img1_region
            
            # Update the panorama
            panorama[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = panorama_region
        else:
            # No overlap, just place image1 using vectorized operations
            valid_start_x = max(0, img1_start_x)
            valid_end_x = min(warped_w, img1_end_x)
            valid_start_y = max(0, img1_start_y)
            valid_end_y = min(warped_h, img1_end_y)
            
            img1_x_start = valid_start_x - offset_x
            img1_x_end = valid_end_x - offset_x
            img1_y_start = valid_start_y - offset_y
            img1_y_end = valid_end_y - offset_y
            
            panorama[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = \
                image1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
        
        return panorama

    def project_to_sphere(self, image):
        """
        Project an image to spherical coordinates using calculated dimensions
        
        Args:
            image: input image (600x400 pixels)
            
        Returns:
            numpy array: spherical projection of the image
        """
        h, w = image.shape[:2]  # Should be 400x600
        
        # Calculate sphere dimensions based on camera parameters
        sphere_width, sphere_height = self.calculate_sphere_dimensions(image)
        
        # Create output spherical image
        sphere_img = np.zeros((sphere_height, sphere_width, image.shape[2]), dtype=image.dtype)
        
        # Create coordinate grids for spherical output
        y_sphere, x_sphere = np.meshgrid(np.arange(sphere_height), np.arange(sphere_width), indexing='ij')
        
        # Convert to normalized coordinates [-1, 1]
        x_norm = (x_sphere - sphere_width/2) / (sphere_width/2)
        y_norm = (y_sphere - sphere_height/2) / (sphere_height/2)
        
        # Convert to spherical coordinates
        # For a 25mm lens, we need to calculate the proper angular coverage
        # Assuming a typical sensor size, we can estimate the field of view
        
        # Calculate field of view based on focal length and sensor size
        # For a typical APS-C sensor (23.5mm width), 25mm focal length gives ~53° FOV
        fov_horizontal = 53.0  # degrees
        fov_vertical = 35.0    # degrees (approximate)
        
        # Convert FOV to radians
        fov_h_rad = np.radians(fov_horizontal)
        fov_v_rad = np.radians(fov_vertical)
        
        # Map normalized coordinates to angular coordinates
        theta = x_norm * fov_h_rad  # Horizontal angle
        phi = y_norm * fov_v_rad    # Vertical angle
        
        # Convert spherical coordinates to image coordinates
        # Using proper perspective projection
        x_img = self.focal_length * np.tan(theta)
        y_img = self.focal_length * np.tan(phi) / np.cos(theta)
        
        # Convert to pixel coordinates
        # Scale to match image dimensions
        x_pixel = x_img * (w / (2 * self.focal_length * np.tan(fov_h_rad/2))) + w/2
        y_pixel = y_img * (h / (2 * self.focal_length * np.tan(fov_v_rad/2))) + h/2
        
        # Use remap for efficient interpolation
        x_pixel = x_pixel.astype(np.float32)
        y_pixel = y_pixel.astype(np.float32)
        
        # Remap the image
        sphere_img = cv2.remap(image, x_pixel, y_pixel, cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return sphere_img

    def calculate_sphere_dimensions(self, image):
        """
        Calculate optimal sphere dimensions based on camera parameters
        
        Args:
            image: input image
            
        Returns:
            tuple: (width, height) for sphere projection
        """
        h, w = image.shape[:2]
        
        # Calculate field of view based on focal length
        # For 25mm focal length on APS-C sensor (23.5mm width)
        sensor_width_mm = 23.5  # APS-C sensor width
        fov_horizontal = 2 * np.arctan(sensor_width_mm / (2 * self.focal_length))
        fov_vertical = 2 * np.arctan((sensor_width_mm * h / w) / (2 * self.focal_length))
        
        # Convert to degrees
        fov_h_deg = np.degrees(fov_horizontal)
        fov_v_deg = np.degrees(fov_vertical)
        
        print(f"Calculated FOV: {fov_h_deg:.1f}° x {fov_v_deg:.1f}°")
        
        # Calculate sphere dimensions based on angular coverage
        # For a single image, we use the full FOV
        # For stitching, we'll adjust based on overlap
        
        # Base sphere dimensions on angular coverage with HIGHER RESOLUTION
        # 360° = full panorama width
        sphere_width = int((fov_h_deg / 360.0) * w * 8)  # Increased scale factor for higher resolution
        sphere_height = int(h * 2)  # Double the height for higher resolution
        
        # Ensure minimum dimensions
        sphere_width = max(sphere_width, w * 2)  # At least double the input width
        sphere_height = max(sphere_height, h * 2)  # At least double the input height
        
        print(f"Calculated sphere dimensions: {sphere_width}x{sphere_height}")
        
        return sphere_width, sphere_height

    def project_images_to_sphere(self, images):
        """
        Project all images to spherical coordinates
        
        Args:
            images: list of input images
            
        Returns:
            list: spherical projections of all images
        """
        if self.projection_type != 'spherical':
            return images  # Return original images for perspective projection
        
        print("Projecting images to spherical coordinates...")
        sphere_images = []
        
        for i, image in enumerate(images):
            print(f"Projecting image {i} to sphere...")
            sphere_img = self.project_to_sphere(image)
            sphere_images.append(sphere_img)
            
            # Visualize if enabled
            if self.visualize:
                self.step_counter += 1
                cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_sphere_projection_{i}.jpg", sphere_img)
                cv2.imshow(f"Spherical Projection {i}", sphere_img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
        
        return sphere_images

    def stitch_intelligent_reference(self, images, reference_index=None):
        """
        Intelligent reference-based stitching that finds the best matches progressively
        
        Args:
            images: list of numpy arrays
            reference_index: index of reference image (defaults to middle image)
            
        Returns:
            numpy array: panorama image
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images for stitching")
        
        # Auto-configure projection parameters if using spherical projection
        if self.projection_type == 'spherical':
            self.auto_configure_projection(images)
        
        # Determine reference image index
        if reference_index is None:
            reference_index = len(images) // 2  # Default to middle image
        elif reference_index < 0 or reference_index >= len(images):
            raise ValueError(f"Invalid reference index: {reference_index}")
        
        print(f"Using image {reference_index} as reference (index {reference_index})")
        
        # Step 1: Project all images to spherical coordinates (if using spherical projection)
        print("Step 1: Projecting images to spherical coordinates...")
        processed_images = self.project_images_to_sphere(images)
        
        # Step 2: Detect features in all SPHERICAL images (after projection)
        print("Step 2: Detecting features in spherical images...")
        all_keypoints = []
        all_descriptors = []
        
        for i, image in enumerate(processed_images):
            print(f"Detecting features in spherical image {i}...")
            keypoints, descriptors = self.detect_features(image)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
        
        # Store for use in lookup methods
        self.all_keypoints = all_keypoints
        self.all_descriptors = all_descriptors
        
        # Step 3: Create correspondence lookup table (EFFICIENT APPROACH)
        print("Step 3: Creating correspondence lookup table...")
        lookup = self.create_correspondence_lookup(all_keypoints, all_descriptors, processed_images)
        
        # Step 4: Start with reference image
        print("Step 4: Starting with reference image...")
        panorama = processed_images[reference_index].copy()
        used_images = {reference_index}
        
        # Step 5: Progressively add images using lookup table (EFFICIENT)
        while len(used_images) < len(images):
            print(f"\nStep 5: Finding next best match using lookup table (used: {len(used_images)}/{len(images)})...")
            
            # Find best next image using lookup table
            best_match_idx, best_match_count, best_homography = self.find_best_next_image(used_images, lookup)
            
            if best_match_idx == -1:
                print("No more images can be stitched!")
                break
            
            print(f"Selected image {best_match_idx} with {best_match_count} total correspondences")
            
            # Step 6: Stitch the best matching image
            print(f"Step 6: Stitching image {best_match_idx} to panorama...")
            
            # For the first few images, we can use the lookup homography directly
            # For later images, we need to re-match against the growing panorama
            if len(used_images) == 1:
                # First image after reference - use lookup homography
                homography = best_homography
            else:
                # Multiple images in panorama - re-match against current panorama
                print(f"Re-matching image {best_match_idx} against current panorama...")
                panorama_keypoints, panorama_descriptors = self.detect_features(panorama)
                
                matches = self.match_features(panorama_descriptors, all_descriptors[best_match_idx], 
                                           panorama, processed_images[best_match_idx], panorama_keypoints, all_keypoints[best_match_idx])
                good_matches = self.filter_matches(matches)
                
                if len(good_matches) >= self.min_matches:
                    homography, mask = self.estimate_homography(panorama_keypoints, all_keypoints[best_match_idx], good_matches)
                    if homography is None:
                        print(f"Failed to estimate homography for image {best_match_idx}")
                        break
                else:
                    print(f"Insufficient matches for image {best_match_idx}")
                    break
            
            # Calculate bounds and warp
            output_width, output_height, offset_x, offset_y = self.calculate_stitching_bounds(
                panorama, processed_images[best_match_idx], homography)
            
            warped = self.warp_image(processed_images[best_match_idx], homography, 
                                   (output_width, output_height), offset_x, offset_y)
            
            if warped is not None:
                # Create new panorama
                panorama = self.create_panorama(panorama, warped, offset_x, offset_y)
                used_images.add(best_match_idx)
                
                print(f"Successfully stitched image {best_match_idx}. Panorama size: {panorama.shape}")
            else:
                print(f"Failed to warp image {best_match_idx}")
                break
        
        # Step 7: Post-process final panorama
        print("Step 7: Post-processing final panorama...")
        final_panorama = self.post_process(panorama)
        
        return final_panorama

    def post_process(self, panorama):
        """Apply minimal post-processing - only crop black borders"""
        if panorama is None:
            return None
        
        # Only crop black borders - no color transformations or filters
        cropped = self.crop_black_borders(panorama)
        
        return cropped

    def crop_black_borders(self, panorama):
        """Crop black borders from the panorama"""
        if panorama is None:
            return None
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
        # Use a higher threshold to be more aggressive about cropping
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find non-zero pixels (non-black regions)
        coords = cv2.findNonZero(thresh)
        
        if coords is None:
            return panorama  # No non-black pixels found
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add small padding to avoid cutting too close to content
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, panorama.shape[1] - x)
        h = min(h + 2*padding, panorama.shape[0] - y)
        
        # Crop the image
        cropped = panorama[y:y+h, x:x+w]
        
        # Ensure we don't return an empty image
        if cropped.size == 0:
            return panorama
        
        return cropped

    def validate_stitching(self, image1, image2, homography):
        """Validate if stitching is likely to succeed"""
        if homography is None:
            return False, 0.0
        
        # Check if homography is reasonable
        det = np.linalg.det(homography)
        if det <= 0:
            return False, 0.0
        
        # If images are provided, check if transformation is not too extreme
        if image1 is not None and image2 is not None:
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            
            # Transform corners of image2
            corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            warped_corners2 = cv2.perspectiveTransform(corners2, homography)
            
            # Check if transformed corners are reasonable
            for corner in warped_corners2:
                x, y = corner.ravel()
                # Check if corner is within reasonable bounds
                if x < -w1 or x > 2*w1 or y < -h1 or y > 2*h1:
                    return False, 0.0
        
        # Calculate confidence based on determinant
        confidence = min(1.0, det / 10.0)  # Normalize determinant
        
        return True, confidence

    def visualize_matches(self, image1, image2, keypoints1, keypoints2, matches):
        """Create visualization of feature matches"""
        # Draw matches
        img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                                     matches, None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches

    def visualize_keypoints(self, image, keypoints, style="rich"):
        """Visualize keypoints on an image"""
        if style == "simple":
            return cv2.drawKeypoints(image, keypoints, None)
        elif style == "rich":
            return cv2.drawKeypoints(image, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        elif style == "color":
            # Custom colored keypoints
            result = image.copy()
            for kp in keypoints:
                x, y = kp.pt
                cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
            return result
        else:
            return cv2.drawKeypoints(image, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def auto_determine_focal_length(self, images):
        """
        Automatically determine focal length from images
        
        Args:
            images: list of input images
            
        Returns:
            float: estimated focal length
        """
        # Method 1: Try to extract from EXIF
        focal_lengths = []
        for i, image in enumerate(images):
            # For now, we'll use a simple estimation
            # In practice, you'd extract EXIF data here
            pass
        
        # Method 2: Estimate from feature geometry
        if not focal_lengths:
            focal_lengths = self.estimate_focal_from_features(images)
        
        # Method 3: Use default based on image analysis
        if not focal_lengths:
            # Analyze image overlap patterns
            overlap_ratio = self.analyze_overlap_ratio(images)
            
            # Estimate focal length based on overlap
            # More overlap = shorter focal length
            if overlap_ratio > 0.5:
                focal_lengths = [35.0]  # Wide angle
            elif overlap_ratio > 0.3:
                focal_lengths = [50.0]  # Normal
            else:
                focal_lengths = [25.0]  # Telephoto
        
        return np.median(focal_lengths) if focal_lengths else 25.0

    def estimate_focal_from_features(self, images):
        """Estimate focal length from feature point analysis"""
        focal_lengths = []
        
        # For spherical projection, we'll estimate focal length differently
        # since we don't want to detect features before projection
        if self.projection_type == 'spherical':
            # Use a simpler estimation based on image analysis
            # This avoids detecting features before projection
            return [self.focal_length]  # Use configured focal length
        
        # For perspective projection, use feature-based estimation
        for i in range(len(images)-1):
            # Detect features in consecutive images
            kp1, des1 = self.detect_features(images[i])
            kp2, des2 = self.detect_features(images[i+1])
            
            if len(kp1) > 10 and len(kp2) > 10:
                # Match features
                matches = self.match_features(des1, des2)
                good_matches = self.filter_matches(matches)
                
                if len(good_matches) > 10:
                    # Analyze feature point distribution
                    # Estimate focal length from geometry
                    focal_length = self.estimate_from_feature_geometry(kp1, kp2, good_matches)
                    if focal_length > 0:
                        focal_lengths.append(focal_length)
        
        return focal_lengths

    def estimate_from_feature_geometry(self, kp1, kp2, matches):
        """Estimate focal length from feature point geometry"""
        if len(matches) < 10:
            return 0
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        
        # Calculate point distances
        distances = np.linalg.norm(src_pts - dst_pts, axis=1)
        
        # Estimate focal length from distance distribution
        # This is a simplified approach - real implementations are more complex
        mean_distance = np.mean(distances)
        
        # Rough estimation based on point spread
        if mean_distance < 50:
            return 35.0  # Wide angle
        elif mean_distance < 100:
            return 50.0  # Normal
        else:
            return 25.0  # Telephoto

    def analyze_overlap_ratio(self, images):
        """Analyze overlap ratio between consecutive images"""
        overlap_ratios = []
        
        # For spherical projection, use a simpler estimation
        # since we don't want to detect features before projection
        if self.projection_type == 'spherical':
            # Estimate overlap based on image content analysis
            # This is a simplified approach for spherical projection
            return 0.4  # Assume medium overlap for spherical projection
        
        # For perspective projection, use feature-based analysis
        for i in range(len(images)-1):
            # Detect features and estimate overlap
            kp1, des1 = self.detect_features(images[i])
            kp2, des2 = self.detect_features(images[i+1])
            
            if len(kp1) > 10 and len(kp2) > 10:
                matches = self.match_features(des1, des2)
                good_matches = self.filter_matches(matches)
                
                # Calculate overlap ratio
                overlap_ratio = len(good_matches) / min(len(kp1), len(kp2))
                overlap_ratios.append(overlap_ratio)
        
        return np.mean(overlap_ratios) if overlap_ratios else 0.3

    def auto_determine_sphere_dimensions(self, images, focal_length):
        """
        Automatically determine optimal sphere dimensions based on camera parameters
        
        Args:
            images: list of input images
            focal_length: estimated focal length
            
        Returns:
            tuple: (width, height) for sphere projection
        """
        # Use the first image to calculate base dimensions
        base_image = images[0]
        sphere_width, sphere_height = self.calculate_sphere_dimensions(base_image)
        
        # Analyze overlap to adjust for panorama width
        overlap_ratio = self.analyze_overlap_ratio(images)
        
        # Adjust width based on overlap (more overlap = narrower panorama needed)
        if overlap_ratio > 0.5:
            # High overlap - narrow panorama
            sphere_width = int(sphere_width * 1.5)
        elif overlap_ratio > 0.3:
            # Medium overlap - standard panorama
            sphere_width = int(sphere_width * 2.0)
        else:
            # Low overlap - wide panorama
            sphere_width = int(sphere_width * 3.0)
        
        return sphere_width, sphere_height

    def auto_configure_projection(self, images):
        """
        Automatically configure projection parameters
        
        Args:
            images: list of input images
            
        Returns:
            dict: configuration parameters
        """
        print("Auto-configuring projection parameters...")
        
        # Step 1: Determine focal length
        focal_length = self.auto_determine_focal_length(images)
        print(f"Estimated focal length: {focal_length:.1f}mm")
        
        # Step 2: Calculate sphere dimensions based on camera parameters
        sphere_width, sphere_height = self.auto_determine_sphere_dimensions(images, focal_length)
        print(f"Calculated sphere dimensions: {sphere_width}x{sphere_height}")
        
        # Step 3: Update configuration
        self.focal_length = focal_length
        
        return {
            'focal_length': focal_length,
            'sphere_width': sphere_width,
            'sphere_height': sphere_height
        }

    def create_correspondence_lookup(self, all_keypoints, all_descriptors, images):
        """
        Create a lookup table of correspondences between all image pairs
        
        Args:
            all_keypoints: list of keypoints for all images
            all_descriptors: list of descriptors for all images
            images: list of input images (for validation)
            
        Returns:
            dict: lookup table with (img1, img2) -> (match_count, homography)
        """
        print("Creating correspondence lookup table...")
        lookup = {}
        
        # Calculate correspondences between all image pairs
        for i in range(len(all_keypoints)):
            for j in range(i + 1, len(all_keypoints)):
                print(f"Calculating correspondences between images {i} and {j}...")
                
                # Match features between image pair
                matches = self.match_features(all_descriptors[i], all_descriptors[j])
                good_matches = self.filter_matches(matches)
                
                if len(good_matches) >= self.min_matches:
                    # Estimate homography
                    homography, mask = self.estimate_homography(all_keypoints[i], all_keypoints[j], good_matches)
                    
                    if homography is not None:
                        # Validate homography
                        is_valid, confidence = self.validate_stitching(images[i], images[j], homography)
                        if is_valid:
                            lookup[(i, j)] = (len(good_matches), homography)
                            lookup[(j, i)] = (len(good_matches), np.linalg.inv(homography))  # Inverse for reverse direction
                            print(f"  Images {i}-{j}: {len(good_matches)} matches")
                        else:
                            print(f"  Images {i}-{j}: homography validation failed")
                    else:
                        print(f"  Images {i}-{j}: homography estimation failed")
                else:
                    print(f"  Images {i}-{j}: insufficient matches ({len(good_matches)} < {self.min_matches})")
        
        return lookup

    def find_best_next_image(self, used_images, lookup):
        """
        Find the best next image to stitch based on correspondence lookup
        
        Args:
            used_images: set of already used image indices
            lookup: correspondence lookup table
            
        Returns:
            tuple: (best_image_idx, best_match_count, best_homography)
        """
        best_image_idx = -1
        best_match_count = 0
        best_homography = None
        
        # Find the remaining image with most correspondences to any used image
        for remaining_img in range(len(self.all_keypoints)):
            if remaining_img in used_images:
                continue
            
            # Check correspondences with all used images
            total_matches = 0
            best_homography_for_img = None
            
            for used_img in used_images:
                key = (used_img, remaining_img)
                if key in lookup:
                    match_count, homography = lookup[key]
                    total_matches += match_count
                    if best_homography_for_img is None or match_count > best_match_count:
                        best_homography_for_img = homography
            
            # Update best if this image has more total correspondences
            if total_matches > best_match_count:
                best_match_count = total_matches
                best_image_idx = remaining_img
                best_homography = best_homography_for_img
        
        return best_image_idx, best_match_count, best_homography

def main():
    """Main function to demonstrate usage"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Manual Image Stitcher')
    parser.add_argument('--dataset', type=str,
                       help='Dataset name (default: boat)')
    parser.add_argument('--output_name', type=str, default='panorama.jpg',
                       help='Name of output panorama file')
    parser.add_argument('--reference_index', type=int, default=None,
                       help='Index of reference image (defaults to middle image)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable step-by-step visualization (enabled by default)')
    parser.add_argument('--projection', type=str,
                       choices=['perspective', 'spherical'],
                       help='Projection type: perspective or spherical (default: perspective)')
    parser.add_argument('--focal_length', type=float, default=25.0,
                       help='Focal length for spherical projection in mm (default: 25.0)')
    # Remove sphere_width and sphere_height arguments - they're calculated automatically
    
    args = parser.parse_args()
    
    # Handle visualization flags
    visualize = not args.no_visualize
    
    # Get paths based on dataset
    input_dir = get_default_image_dir(args.dataset)
    output_dir = get_output_dir(args.dataset, "output")
    steps_dir = get_output_dir(args.dataset, "steps")
    
    # Initialize stitcher with projection settings
    config = StitchingConfig()
    config.projection_type = args.projection
    config.focal_length = args.focal_length
    
    stitcher = ImageStitcher(config, visualize=visualize, test_output_dir=steps_dir)
    
    # Load images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.JPG")))
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    images = []
    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)} - Shape: {img.shape}")
    
    if len(images) < 2:
        print("Need at least 2 images for stitching")
        return
    
    # Stitch panorama using intelligent reference-based approach
    panorama = stitcher.stitch_intelligent_reference(images, args.reference_index)
    
    if panorama is not None:
        # Save result with projection type in filename
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(args.output_name)[0]
        ext = os.path.splitext(args.output_name)[1]
        output_filename = f"{base_name}_{args.projection}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved as {output_path} using {args.projection} projection")
        
        # Display result
        cv2.imshow(f"Panorama - {args.projection}", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed")

if __name__ == "__main__":
    main() 
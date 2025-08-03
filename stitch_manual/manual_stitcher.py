import cv2
import numpy as np
import os
import glob
from pathlib import Path

class StitchingConfig:
    """Configuration class for stitching parameters"""
    def __init__(self):
        self.feature_detector = 'SIFT'  # or 'SURF', 'ORB'
        self.matcher_type = 'FLANN'     # or 'BF'
        self.match_ratio = 0.75
        self.ransac_threshold = 5.0
        self.blend_width = 50
        self.min_matches = 10

class ManualImageStitcher:
    """Manual image stitching implementation with step-by-step control"""
    
    def __init__(self, config=None, visualize=False):
        """
        Initialize the manual stitcher
        
        Args:
            config: StitchingConfig object with parameters
            visualize: if True, shows intermediate steps
        """
        self.config = config or StitchingConfig()
        self.visualize = visualize
        self.step_counter = 0  # Track visualization steps
        self.test_output_dir = "stitch_manual/test_output"
        
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
    

    def detect_features(self, image):
        """
        Detect keypoints and extract descriptors from a single image
        
        Args:
            image: numpy array of the input image
            
        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: list of cv2.KeyPoint objects
                - descriptors: numpy array of feature descriptors
        """
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
        """
        Match features between two sets of descriptors
        
        Args:
            descriptors1: numpy array of descriptors from first image
            descriptors2: numpy array of descriptors from second image
            image1: first image (for visualization)
            image2: second image (for visualization)
            keypoints1: keypoints from first image (for visualization)
            keypoints2: keypoints from second image (for visualization)
            
        Returns:
            list: list of cv2.DMatch objects (all matches, not filtered)
        """
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
        """
        Filter matches using Lowe's ratio test
        
        Args:
            matches: list of cv2.DMatch objects (from knnMatch with k=2)
            
        Returns:
            list: filtered matches
        """
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
        """
        Estimate homography matrix between two sets of keypoints
        
        Args:
            keypoints1: list of cv2.KeyPoint objects from first image
            keypoints2: list of cv2.KeyPoint objects from second image
            matches: list of cv2.DMatch objects
            
        Returns:
            tuple: (homography_matrix, mask) or (None, None) if failed
                - homography_matrix: 3x3 numpy array
                - mask: boolean array indicating inliers
        """
        if len(matches) < self.min_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            return None, None
        
        # Extract matched keypoints
        # We want to warp image2 to align with image1, so:
        # src_pts = points in image2 (to be warped)
        # dst_pts = corresponding points in image1 (reference)
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
        
        # Additional validation: check if homography is reasonable
        try:
            # Check determinant (should be positive for valid transformation)
            det = np.linalg.det(H)
            if det <= 0:
                print("Warning: Homography determinant is non-positive")
                return None, None
            
            # Check if transformation is not too extreme
            if abs(det) > 1000 or abs(det) < 0.001:
                print("Warning: Homography determinant is extreme, transformation may be unreliable")
                return None, None
                
        except np.linalg.LinAlgError:
            print("Error: Homography matrix is singular")
            return None, None
        
        # Visualize if enabled
        if self.visualize and hasattr(self, '_last_images'):
            self.step_counter += 1
            image1, image2 = self._last_images
            img_homography = self.visualize_homography(image1, image2, H)
            
            # Save visualization
            cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_homography.jpg", img_homography)
            
            cv2.imshow("Homography Transformation", img_homography)
            cv2.waitKey(500)  # Wait 0.5 seconds for better flow
            cv2.destroyAllWindows()
            
        return H, mask
    
    def warp_image(self, image, homography, output_size, offset_x=0, offset_y=0):
        """
        Warp an image using homography matrix with optional offset
        
        Args:
            image: numpy array of input image
            homography: 3x3 homography matrix
            output_size: tuple (width, height) of output image
            offset_x: x offset for translation
            offset_y: y offset for translation
            
        Returns:
            numpy array: warped image
        """
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
    
    def calculate_stitching_bounds(self, image1, image2, homography):
        """
        Calculate the bounds for the stitched panorama
        
        Args:
            image1: first image (reference image)
            image2: second image (image to be warped)
            homography: homography matrix from image2 to image1
            
        Returns:
            tuple: (output_width, output_height, offset_x, offset_y)
        """
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
        
        # Debug: Print more detailed information
        print(f"Bounds calculation:")
        print(f"  Image1 corners: {corners1}")
        print(f"  Warped image2 corners: {warped_corners2.reshape(-1, 2)}")
        print(f"  Min/Max X: {min_x}/{max_x}, Min/Max Y: {min_y}/{max_y}")
        print(f"  Output size: {output_width}x{output_height}")
        print(f"  Offset: ({offset_x}, {offset_y})")
        
        # Additional validation: check if the warped image corners make sense
        warped_corners_flat = warped_corners2.reshape(-1, 2)
        print(f"  Warped image2 Y coordinates: {warped_corners_flat[:, 1]}")
        print(f"  Image1 Y coordinates: {corners1[:, 1]}")
        
        return output_width, output_height, offset_x, offset_y
        
    
    def create_panorama(self, image1, warped_image, offset_x, offset_y):
        """
        Create panorama by combining image1 and warped image with smooth transition
        
        Args:
            image1: first image (reference image)
            warped_image: second image after warping (already positioned correctly)
            offset_x: x offset for placing image1
            offset_y: y offset for placing image1
            
        Returns:
            numpy array: combined panorama
        """
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
                
                # Vectorized blending
                # Check if warped image has content (not black)
                warped_has_content = np.any(panorama_region > 10, axis=2, keepdims=True)
                
                # Blend where warped image has content
                blended = (alpha_mask * img1_region + 
                          (1 - alpha_mask) * panorama_region).astype(np.uint8)
                
                # Use blended result where warped has content, otherwise use image1
                panorama_region = np.where(warped_has_content, blended, img1_region)
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

    def stitch_pair(self, image1, image2):
        """
        Complete pipeline for stitching two images
        
        Args:
            image1: first image
            image2: second image
            
        Returns:
            numpy array: stitched image
        """
        if self.visualize:
            print("=== Step-by-step visualization ===")
        
        # Store images for visualization
        if self.visualize:
            self._last_images = (image1, image2)
        
        # Step 1: Detect features
        if self.visualize:
            print("Step 1: Detecting features...")
        
        keypoints1, descriptors1 = self.detect_features(image1)
        keypoints2, descriptors2 = self.detect_features(image2)
        
        # Step 2: Match features
        if self.visualize:
            print("Step 2: Matching features...")
        
        matches = self.match_features(descriptors1, descriptors2, image1, image2, keypoints1, keypoints2)
        good_matches = self.filter_matches(matches)
        
        # Step 3: Estimate homography
        if self.visualize:
            print("Step 3: Estimating homography...")
        
        homography, mask = self.estimate_homography(keypoints1, keypoints2, good_matches)
        
        if homography is None:
            print("Failed to estimate homography")
            return None
        
        # Validate the homography
        is_valid, confidence = self.validate_stitching(image1, image2, homography)
        if not is_valid:
            print(f"Homography validation failed (confidence: {confidence:.2f})")
            return None
        else:
            print(f"Homography validation passed (confidence: {confidence:.2f})")
        
        # Step 4: Calculate bounds and warp
        if self.visualize:
            print("Step 4: Warping image...")
        
        output_width, output_height, offset_x, offset_y = self.calculate_stitching_bounds(
            image1, image2, homography)
        
        print(f"Panorama bounds: {output_width}x{output_height}, offset: ({offset_x}, {offset_y})")
        
        warped = self.warp_image(image2, homography, (output_width, output_height), offset_x, offset_y)
        
        # Step 5: Create panorama
        if self.visualize:
            print("Step 5: Creating panorama...")
        
        panorama = self.create_panorama(image1, warped, offset_x, offset_y)
        
        # Step 6: Post-process
        if self.visualize:
            print("Step 6: Post-processing...")
        
        final_panorama = self.post_process(panorama)
        
        # Visualize final result if enabled
        if self.visualize:
            self.step_counter += 1
            
            # Save visualization
            cv2.imwrite(f"{self.test_output_dir}/step_{self.step_counter:02d}_final_panorama.jpg", final_panorama)
            
            cv2.imshow("Final Panorama", final_panorama)
            cv2.waitKey(1000)  # Wait 1 second for final result
            cv2.destroyAllWindows()
        
        return final_panorama

    def stitch_multiple(self, images):
        """
        Stitch multiple images sequentially
        
        Args:
            images: list of numpy arrays
            
        Returns:
            numpy array: panorama image
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images for stitching")
        
        # Start with first image
        panorama = images[0]
        
        # Stitch each subsequent image
        for i in range(1, len(images)):
            panorama = self.stitch_pair(panorama, images[i])
            if panorama is None:
                print(f"Failed to stitch image {i+1}")
                return None
        
        return panorama
    
    def crop_black_borders(self, panorama):
        """
        Crop black borders from the panorama
        
        Args:
            panorama: input panorama image
            
        Returns:
            numpy array: cropped panorama
        """
        if panorama is None:
            return None
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
        # Use a higher threshold to be more aggressive about cropping
        # This helps remove dark borders that might not be pure black
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
    
    def post_process(self, panorama):
        """
        Apply minimal post-processing - only crop black borders
        
        Args:
            panorama: input panorama image
            
        Returns:
            numpy array: processed panorama
        """
        if panorama is None:
            return None
        
        # Only crop black borders - no color transformations or filters
        cropped = self.crop_black_borders(panorama)
        
        return cropped
    
    def visualize_matches(self, image1, image2, keypoints1, keypoints2, matches):
        """
        Create visualization of feature matches
        
        Args:
            image1: first image
            image2: second image
            keypoints1: keypoints from first image
            keypoints2: keypoints from second image
            matches: list of cv2.DMatch objects
            
        Returns:
            numpy array: visualization image
        """
        # Draw matches
        img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                                     matches, None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches
    
    def visualize_keypoints(self, image, keypoints, style="rich"):
        """
        Visualize keypoints on an image
        
        Args:
            image: input image
            keypoints: list of cv2.KeyPoint objects
            style: visualization style ("simple", "rich", "color")
            
        Returns:
            numpy array: image with keypoints drawn
        """
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
    
    def visualize_homography(self, image1, image2, homography, title="Homography"):
        """
        Visualize homography transformation
        
        Args:
            image1: first image
            image2: second image
            homography: homography matrix
            title: window title
            
        Returns:
            numpy array: visualization image
        """
        # Get image dimensions
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Transform corners of image2
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, homography)
        
        # Create a visualization showing both images and the transformation
        # Create a larger canvas to show both images
        canvas_width = w1 + w2
        canvas_height = max(h1, h2)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place image1 on the left
        canvas[:h1, :w1] = image1
        
        # Place image2 on the right
        canvas[:h2, w1:w1+w2] = image2
        
        # Draw the transformed corners on the canvas
        for i, corner in enumerate(warped_corners2):
            x, y = corner.ravel()
            # Draw corner on the left side (image1 area)
            cv2.circle(canvas, (int(x), int(y)), 15, (0, 255, 0), 3)
            # Add corner labels
            cv2.putText(canvas, str(i+1), (int(x)+20, int(y)+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw original corners of image2 (on the right side)
        original_corners = [(w1, 0), (w1+w2, 0), (w1+w2, h2), (w1, h2)]
        for i, (x, y) in enumerate(original_corners):
            cv2.circle(canvas, (x, y), 10, (255, 0, 0), 2)
            cv2.putText(canvas, f"{i+1}'", (x+15, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add connecting lines to show the transformation
        for i in range(4):
            x1, y1 = original_corners[i]
            x2, y2 = warped_corners2[i].ravel()
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        
        return canvas
    
    def validate_stitching(self, image1, image2, homography):
        """
        Validate if stitching is likely to succeed
        
        Args:
            image1: first image
            image2: second image
            homography: homography matrix
            
        Returns:
            tuple: (is_valid, confidence_score)
        """
        if homography is None:
            return False, 0.0
        
        # Check if homography is reasonable
        # 1. Check determinant (should be positive for valid transformation)
        det = np.linalg.det(homography)
        if det <= 0:
            return False, 0.0
        
        # 2. Check if transformation is not too extreme
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
        
        # Calculate confidence based on determinant and corner positions
        confidence = min(1.0, det / 10.0)  # Normalize determinant
        
        return True, confidence
    
    def stitch_panorama(self, images):
        """
        Complete end-to-end stitching pipeline
        
        Args:
            images: list of numpy arrays
            
        Returns:
            numpy array: final panorama
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images for stitching")
        
        # Step 1: Stitch multiple images
        panorama = self.stitch_multiple(images)
        
        if panorama is None:
            return None
        
        # Step 2: Post-process the result
        processed_panorama = self.post_process(panorama)
        
        return processed_panorama

def main():
    """Main function to demonstrate usage"""
    # Initialize stitcher
    config = StitchingConfig()
    stitcher = ManualImageStitcher(config)
    
    # Load images
    image_dir = "../images"
    image_files = glob.glob(os.path.join(image_dir, "*.JPG"))
    image_files.sort()
    
    images = []
    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)} - Shape: {img.shape}")
    
    if len(images) >= 2:
        # Stitch panorama
        panorama = stitcher.stitch_panorama(images)
        
        if panorama is not None:
            # Save result
            os.makedirs("output", exist_ok=True)
            cv2.imwrite("output/manual_panorama.jpg", panorama)
            print("Manual panorama saved as output/manual_panorama.jpg")
            
            # Display result
            cv2.imshow("Manual Panorama", panorama)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Manual stitching failed")
    else:
        print("Need at least 2 images for stitching")

if __name__ == "__main__":
    main() 
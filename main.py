#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
import random

def load_data(data_dir):
    # Load the first image
    img1 = cv2.imread(os.path.join(data_dir, 'I1.png'))
    if img1 is None:
        raise ValueError("Failed to load the first image")
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 1 data type: {img1.dtype}")
    
    # Load the second image
    img2 = cv2.imread(os.path.join(data_dir, 'I2.png'))
    if img2 is None:
        raise ValueError("Failed to load the second image")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Image 2 data type: {img2.dtype}")
    
    # Load the camera calibration matrix from a file
    K_file = os.path.join(data_dir, 'K.txt')
    K = load_camera_matrix(K_file)
    
    return img1, img2, K

def load_camera_matrix(K_file):
    K = np.loadtxt(K_file, delimiter=',')
    return K

def find_interest_points(img):
    # Use SIFT to detect and compute the keypoints and their descriptors
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    return kp, des

def visualize_interest_points(img1, img2, kp1, kp2):
    # Create a copy of the images to draw the keypoints on
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    
    # Concatenate the images horizontally for visualization
    result = cv2.hconcat([img1_kp, img2_kp])
    
    # Display the result using Matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(result[:, :, ::-1])
    plt.title('Interest Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def match_interest_points(img1, img2, kp1, kp2, des1, des2):
    # Use BFMatcher to match the keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter out poor matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Select a random subset of 70 matches if there are enough matches
    if len(good_matches) > 70:
        good_matches = random.sample(good_matches, 70)

    # Extract the keypoint coordinates for the good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Image with only good matched keypoints
    keypoints_only_img1 = cv2.drawKeypoints(img1, [kp1[m.queryIdx] for m in good_matches], None, color=(0, 255, 0))
    keypoints_only_img2 = cv2.drawKeypoints(img2, [kp2[m.trainIdx] for m in good_matches], None, color=(0, 255, 0))
    keypoints_only_img = cv2.hconcat([keypoints_only_img1, keypoints_only_img2])

    # Image with thin lines connecting good matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchColor=(255, 0, 0), flags=2)

    # Visualize both images
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints_only_img[:, :, ::-1])
    plt.title('Keypoints Only')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches[:, :, ::-1])
    plt.title('Matched Interest Points with Lines')
    plt.axis('off')
    plt.show()

    return good_matches, pts1, pts2



def main():
    # Set the data directory
    data_dir = 'data/example_1'
    
    # Load the input data
    img1, img2, K = load_data(data_dir)
    
    # Step 1: Find interest points
    kp1, des1 = find_interest_points(img1)
    kp2, des2 = find_interest_points(img2)
    
    # Visualize the interest points
    visualize_interest_points(img1, img2, kp1, kp2)
    
      # Step 2: Match interest points
    good_matches, pts1, pts2 = match_interest_points(img1, img2, kp1, kp2, des1, des2)    

    

    
    # Step 5: Estimate the Essential matrix
    # (Implement this step later)
    
    # Step 6: 3D reconstruction
    # (Implement this step later)
    
    # Step 7: Detect planes
    # (Implement this step later)
    
    # Step 8: Visualize the normals
    # (Implement this step later)
    
    # Save the results
    # (Implement this step later)

if __name__ == '__main__':
    main()
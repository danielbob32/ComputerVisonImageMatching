#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
import random

def select_random_matches(matches, num_samples=50):
    # Ensure you don't try to sample more matches than exist
    num_samples = min(len(matches), num_samples)
    selected_matches = np.random.choice(matches, num_samples, replace=False)
    return selected_matches

def load_data(data_dir):
    # Load the first image
    img1 = cv2.imread(os.path.join(data_dir, 'I1.png'))
    if img1 is None:
        raise ValueError("Failed to load the first image")

    
    # Load the second image
    img2 = cv2.imread(os.path.join(data_dir, 'I2.png'))
    if img2 is None:
        raise ValueError("Failed to load the second image")

    
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

def validate_epipolar_constraint(pts1, pts2, F):
    ones = np.ones((pts1.shape[0], 1))
    pts1_hom = np.hstack([pts1, ones])  # Convert to homogeneous coordinates
    pts2_hom = np.hstack([pts2, ones])

    # Compute the epipolar lines for pts1
    lines = np.dot(F, pts1_hom.T).T  # Nx3

    # Evaluate the epipolar constraint: pts2_hom * line = 0
    errors = np.abs(np.sum(lines * pts2_hom, axis=1))
    return errors



def visualize_interest_points(img1, img2, kp1, kp2):
    # Create a copy of the images to draw the keypoints on
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 0, 255))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 0, 255))
    
    # Concatenate the images horizontally for visualization
    result = cv2.hconcat([img1_kp, img2_kp])
    
    # Display the result using Matplotlib
    plt.figure(figsize=(20, 10))
    plt.imshow(result[:, :, ::-1])
    plt.title('Interest Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    # Assuming pts1 and pts2 are the corresponding points in each image
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines = draw_lines_e(img1, lines1, pts1)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines = draw_lines_e(img2, lines2, pts2)

    return img1_epilines, img2_epilines

def draw_lines_e(img, lines, pts):
    ''' img - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img.shape
    for r, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
        # Ensure pt coordinates are integers
        pt = (int(pt[0][0]), int(pt[0][1]))  # Corrected the point format here
        img = cv2.circle(img, pt, 5, color, -1)
    return img

def draw_lines(img, lines, pts):
    ''' img - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img.shape
    for r, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv2.circle(img, tuple(pt[0]), 5, color, -1)
    return img
 
def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
    """Draw a dotted line in img from pt1 to pt2 with given color and thickness."""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    points = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        points.append((x, y))
    for point in points:
        cv2.circle(img, point, thickness, color, -1)  # Draw filled circle (dot) at each point
        
def draw_custom_matches(img1, img2, kp1, kp2, matches):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    height = max(h1, h2)
    width = w1 + w2
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    output_image[:h1, :w1] = img1
    output_image[:h2, w1:w1+w2] = img2

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt
        
        cv2.circle(output_image, (int(x1), int(y1)), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(output_image, (int(x2 + w1), int(y2)), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        
        draw_dotted_line(output_image, (int(x1), int(y1)), (int(x2 + w1), int(y2)), (0, 255, 255), thickness=1, gap=5)

    return output_image
   
def visualize_matched_points(img1, img2, kp1, kp2, matches, color1, color2, title):
    # Create a copy of the images to draw the keypoints on
    img1_kp = cv2.drawKeypoints(img1, [kp1[m.queryIdx] for m in matches], None, color=color2)
    img2_kp = cv2.drawKeypoints(img2, [kp2[m.trainIdx] for m in matches], None, color=color2)
    
    # Draw lines connecting the matched keypoints left keypoint in red, right keypoint in blue
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(255, 0, 0), flags=2)
    
    
    # Concatenate the images horizontally for visualization
    result = cv2.hconcat([img1_kp, img2_kp])
    
    # Display the result using Matplotlib
    plt.figure(figsize=(20, 10))
    plt.imshow(result[:, :, ::-1])
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    output_image = draw_custom_matches(img1, img2, kp1, kp2, matches)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(title + ' with Lines')
    plt.axis('off')
    plt.show()
    
def visualize_matches(img1, img2, kp1, kp2, matches, color1, color2, title):
    # Create a copy of the images to draw the keypoints on
    img1_kp = cv2.drawKeypoints(img1, [kp1[m.queryIdx] for m in matches], None, color=color2)
    img2_kp = cv2.drawKeypoints(img2, [kp2[m.trainIdx] for m in matches], None, color=color1)
    
    # Draw lines connecting the matched keypoints
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(255, 0, 0), flags=2)
    
    # Concatenate the images horizontally for visualization
    result = cv2.hconcat([img1_kp, img2_kp])
    
    # Display the result using Matplotlib
    output_image = draw_custom_matches(img1, img2, kp1, kp2, matches)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(title + ' with Lines')
    plt.axis('off')
    plt.show()

def visualize_epipolar_lines(img1, img2, kp1, kp2, pts1, pts2, F, title):
    # Create a copy of the first image to draw the epipolar lines
    img3 = np.copy(img1)
    h, w, _ = img1.shape
    
    # Draw the epipolar lines
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        
        # Draw the epipolar line connecting the matched points
        color = tuple(np.random.randint(0, 255, 3).tolist())
        line1 = cv2.line(img3, (0, int(-F[1,2]/F[1,1])), (w, int(-(F[1,2]+F[0,1]*w)/F[1,1])), color, 1)
        line2 = cv2.line(img2, (0, int(-F[1,2]/F[1,1])), (w, int(-(F[1,2]+F[0,1]*w)/F[1,1])), color, 1)
    
    # Display the result using Matplotlib
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.hconcat([img3, img2]), cmap='gray')
    plt.title(title)
    plt.axis('off')
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
            
    # Shuffle the matches to select a random subset
    random.shuffle(good_matches)
    good_matches = good_matches[:200]

    return good_matches


def filter_matches_based_on_E_and_F(kp1, kp2, matches, K):
    # Extract points from the matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Compute the essential matrix from the fundamental matrix
    E = K.T @ F @ K

    # Filter matches using the mask
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    return inlier_matches, E, F

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
    
    # Step 2: Match interest points (without filtering)
    matches = match_interest_points(img1, img2, kp1, kp2, des1, des2)
    
    # Visualize the matched interest points (keypoints only, before filtering)
    visualize_matched_points(img1, img2, kp1, kp2, matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points (Before Filtering)')
    
    
    # Step 3: Filter the Matches
    inlier_matches, F, E = filter_matches_based_on_E_and_F(kp1, kp2, matches, K)
    
    # Print E and F
    print("Fundamental matrix:")
    print(F)
    
    print("Essential matrix:")
    print(E)
    
    # Visualize the matched interest points (keypoints + lines, after filtering)
    visualize_matches(img1, img2, kp1, kp2, inlier_matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points with Lines (After Filtering)')
    
    # Visualize a random subset of inlier matches with epipolar lines
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
    img1_epilines, img2_epilines = draw_epipolar_lines(img1, img2, pts1, pts2, F)
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(img1_epilines)
    plt.subplot(122), plt.imshow(img2_epilines)
    plt.show()
    
    
    

    
    # Step 5: 3D reconstruction
    # (Implement this step later)
    
    # Step 6: Detect planes
    # (Implement this step later)
    
    # Step 7: Visualize the normals
    # (Implement this step later)
    
    # Save the results
    # (Implement this step later)

if __name__ == '__main__':
    main()
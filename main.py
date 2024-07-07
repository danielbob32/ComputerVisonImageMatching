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

def normalize_points(pts):
    # Ensure points are in the correct shape (N, 2)
    if pts.ndim > 2 and pts.shape[2] > 1:
        pts = pts.reshape(-1, 2)  # Flatten to (N, 2) if not already

    # Compute the centroid of the points
    centroid = np.mean(pts, axis=0)

    # Compute the average distance of the points from the centroid
    avg_dist = np.mean(np.linalg.norm(pts - centroid, axis=1))

    # Compute the scaling factor
    scale = np.sqrt(2) / avg_dist

    # Create the normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    # Normalize the points
    pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Make pts homogeneous
    pts_normalized = np.dot(T, pts_homogeneous.T).T[:, :2]

    return pts_normalized, T



def compute_essential_matrix(kp1, kp2, matches, K):
    # Extract points from the matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Normalize the points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Compute the essential matrix using the 8-point algorithm
    E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, np.eye(3), cv2.RANSAC, 0.999, 1.0)

    # Denormalize the essential matrix
    E = K.T @ E @ K

    return E, mask
  
def triangulate_points(kp1, kp2, matches, P1, P2):
    # Extract points from the matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Triangulate the points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.transpose(), pts2.transpose())

    # Convert to 3D points
    points_3d = points_4d[:3] / points_4d[3]

    return points_3d.T


def visualize_epipolar_lines(img1, img2, kp1, kp2, matches, K, F, title):


    img1_epilines = np.copy(img1)
    img2_epilines = np.copy(img2)

    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        color = random_color()
        colorcirile = random_color()
        
        # Draw the epipolar line for point p2 in img1
        line1 = np.dot(F, [p2[0], p2[1], 1])
        pt1 = map_line_to_border(line1, img1.shape)
        cv2.line(img1_epilines, pt1[0], pt1[1], color, 2)
        cv2.circle(img1_epilines, (int(p1[0]), int(p1[1])), 5, colorcirile, -1)

        # Draw the epipolar line for point p1 in img2
        line2 = np.dot(F.T, [p1[0], p1[1], 1])
        pt2 = map_line_to_border(line2, img2.shape)
        cv2.line(img2_epilines, pt2[0], pt2[1], color, 2)
        cv2.circle(img2_epilines, (int(p2[0]), int(p2[1])), 5, colorcirile, -1)

    result = cv2.hconcat([img1_epilines, img2_epilines])
    plt.figure(figsize=(20, 10))
    plt.imshow(result[:, :, ::-1])
    plt.title(title)
    plt.axis('off')
    plt.show()


def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def map_line_to_border(line, shape):
    a, b, c = line
    x0, y0, x1, y1 = 0, 0, shape[1], shape[0]  # image borders
    if abs(b) > 1e-5:  # horizontal line check
        y0 = int(-c / b)
        y1 = int(-(c + a * shape[1]) / b)
    if abs(a) > 1e-5:  # vertical line check
        x0 = int(-c / a)
        x1 = int(-(c + b * shape[0]) / a)
    # Clipping line points to image boundaries
    y0, y1 = np.clip([y0, y1], 0, shape[0]-1)
    x0, x1 = np.clip([x0, x1], 0, shape[1]-1)
    return (x0, y0), (x1, y1)



def filter_matches_based_on_E_and_F(kp1, kp2, matches, K):
    # Compute the essential matrix  
    E, mask = compute_essential_matrix(kp1, kp2, matches, K)
    
    # Extract the inlier matches
    inlier_matches = [m for i, m in enumerate(matches) if mask[i] == 1]
    
    # Compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2),
                                     np.float32([kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2),
                                     cv2.FM_LMEDS)
    
    # Extract the inlier matches
    inlier_matches = [m for i, m in enumerate(inlier_matches) if mask[i] == 1]
    
    
    return inlier_matches, E, F, mask



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
        
        cv2.circle(output_image, (int(x1), int(y1)), 7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(output_image, (int(x2 + w1), int(y2)), 7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        
        draw_dotted_line(output_image, (int(x1), int(y1)), (int(x2 + w1), int(y2)), (0, 255, 255), thickness=1, gap=5)

    return output_image
   
def visualize_matched_points(img1, img2, kp1, kp2, matches, color1, color2, title):
    # Create a copy of the images to draw the keypoints on
    img1_kp = cv2.drawKeypoints(img1, [kp1[m.queryIdx] for m in matches], None, color=color2)
    img2_kp = cv2.drawKeypoints(img2, [kp2[m.trainIdx] for m in matches], None, color=color2)
    
    
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

    
def match_interest_points(img1, img2, kp1, kp2, des1, des2, num_matches=2):
    # Use BFMatcher to match the keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter out poor matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Select a random subset of good matches
    if len(good_matches) > num_matches:
        selected_matches = random.sample(good_matches, num_matches)
    else:
        selected_matches = good_matches

    return selected_matches


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
    matches = match_interest_points(img1, img2, kp1, kp2, des1, des2, num_matches=40)  # Now flexible to change
    print("Number of matches:", len(matches))
    # Visualize the matched interest points (keypoints only, before filtering)
    visualize_matched_points(img1, img2, kp1, kp2, matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points (Before Filtering)')
    
    
    # Step 3: Filter the Matches
    inlier_matches, F, E ,mask= filter_matches_based_on_E_and_F(kp1, kp2, matches, K)
    print("Number of inlier matches:", len(inlier_matches))

    
    # Print E and F
    print("Fundamental matrix:")
    print(F)
    
    print("Essential matrix:")
    print(E)
    
    # Visualize the matched interest points (keypoints + lines, after filtering)
    visualize_matches(img1, img2, kp1, kp2, inlier_matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points with Lines (After Filtering)')
    
    # Visualize the epipolar lines
    visualize_epipolar_lines(img1, img2, kp1, kp2,inlier_matches, K, F, title='Epipolar Lines')


if __name__ == '__main__':
    main()
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.widgets as widgets
import open3d as o3d

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
       # Ensure the images have the same number of rows
    if img1.shape[0] != img2.shape[0]:
        print(f"Resizing img2 from {img2.shape} to match img1 {img1.shape}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
     # Ensure the images have the same type
    if img1.dtype != img2.dtype:
        raise ValueError(f"img1 and img2 must have the same type. img1 is {img1.dtype}, img2 is {img2.dtype}.")
    
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

def compute_essential_matrix(kp1, kp2, matches, K):
    # Extract points from the matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute the essential matrix using the 5-point algorithm with RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    return E, mask

def visualize_epipolar_lines(img1, img2, kp1, kp2, matches, F, title):
    img1_epilines = np.copy(img1)
    img2_epilines = np.copy(img2)
    
    # Extract points from matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute epilines for points in the first image and draw them on the second image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F).reshape(-1, 3)

    for r1, r2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
        color = random_color()

        # Draw epiline in the first image
        x0, y0, x1, y1 = map_line_to_border(r1, img1.shape)
        cv2.line(img1_epilines, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img1_epilines, (int(pt1[0][0]), int(pt1[0][1])), 5, color, -1)

        # Draw epiline in the second image
        x0, y0, x1, y1 = map_line_to_border(r2, img2.shape)
        cv2.line(img2_epilines, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img2_epilines, (int(pt2[0][0]), int(pt2[0][1])), 5, color, -1)
    
    # Check for dimension and type match
    if img1_epilines.shape != img2_epilines.shape:
        print(f"Shapes before resizing: img1_epilines: {img1_epilines.shape}, img2_epilines: {img2_epilines.shape}")
        img2_epilines = cv2.resize(img2_epilines, (img1_epilines.shape[1], img1_epilines.shape[0]))
        print(f"Shapes after resizing: img1_epilines: {img1_epilines.shape}, img2_epilines: {img2_epilines.shape}")
        
    if img1_epilines.dtype != img2_epilines.dtype:
        raise ValueError(f"img1_epilines and img2_epilines must have the same type. img1_epilines is {img1_epilines.dtype}, img2_epilines is {img2_epilines.dtype}.")

    result = cv2.hconcat([img1_epilines, img2_epilines])
    plt.figure(figsize=(20, 10))
    plt.imshow(result[:, :, ::-1])
    plt.title(title)
    plt.axis('off')
    plt.show()

def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def map_line_to_border(line, img_shape):
    a, b, c = line
    x0, y0 = 0, int(-c / b)
    x1, y1 = img_shape[1], int(-(c + a * img_shape[1]) / b)
    if y0 < 0:
        y0 = 0
        x0 = int(-c / a)
    if y0 >= img_shape[0]:
        y0 = img_shape[0] - 1
        x0 = int(-(c + b * y0) / a)
    if y1 < 0:
        y1 = 0
        x1 = int(-c / a)
    if y1 >= img_shape[0]:
        y1 = img_shape[0] - 1
        x1 = int(-(c + b * y1) / a)
    return x0, y0, x1, y1

def filter_matches_based_on_E_and_F(kp1, kp2, matches, K):
    # Compute the essential matrix
    E, mask_E = compute_essential_matrix(kp1, kp2, matches, K)
    
    # Extract the inlier matches based on the essential matrix
    inlier_matches = [m for i, m in enumerate(matches) if mask_E[i] == 1]
    
    # Extract points from the inlier matches
    pts1_inliers = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
    pts2_inliers = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
    
    # Compute the fundamental matrix using the inlier points
    F, mask_F = cv2.findFundamentalMat(pts1_inliers, pts2_inliers, method=cv2.FM_LMEDS)
    
    # Extract the final inlier matches based on the fundamental matrix
    final_inlier_matches = [m for i, m in enumerate(inlier_matches) if mask_F[i] == 1]
    
    return final_inlier_matches, E, F


def match_interest_points(des1, des2, ratio_test, num_matches):
    # Use BFMatcher to match the keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter out poor matches
    good_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]

    if len(good_matches) > num_matches:
        print(f"Number of matches ({len(good_matches)}) exceeds the limit ({num_matches}). Randomly selecting {num_matches} matches.")
        selected_matches = random.sample(good_matches, num_matches)
    else:
        selected_matches = good_matches

    return selected_matches

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
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
    # Ensure the images have the same number of rows
    if img1.shape[0] != img2.shape[0]:
        print(f"Resizing img2 from {img2.shape} to match img1 {img1.shape}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Ensure the images have the same type
    if img1.dtype != img2.dtype:
        raise ValueError(f"img1 and img2 must have the same type. img1 is {img1.dtype}, img2 is {img2.dtype}.")
    
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
    
    # Display the result using Matplotlib
    output_image = draw_custom_matches(img1, img2, kp1, kp2, matches)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(title + ' with Lines')
    plt.axis('off')
    plt.show()

def decompose_essential_matrix(E, K):
    # Decompose the essential matrix into rotation and translation
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Compute the camera projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1  # Apply intrinsic matrix

    P2_list = []
    for R in [R1, R2]:
        for sign in [1, -1]:
            P2 = np.hstack((R, sign * t))
            P2 = K @ P2  # Apply intrinsic matrix
            P2_list.append(P2)

    return P1, P2_list

def triangulate_points(kp1, kp2, P1, P2, inlier_matches):
    # Convert keypoints to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
    
    # Triangulate points
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous coordinates to 3D
    pts3D = pts4D[:3] / pts4D[3]
    
    # Debugging output: Check for points behind the camera
    num_negative_depths = np.sum(pts3D[2] < 0)
    print(f"Number of points with negative depth: {num_negative_depths}")

    # Check if any points are NaN or Inf and filter them out
    valid_indices = np.all(np.isfinite(pts3D), axis=0)
    pts3D = pts3D[:, valid_indices]
    
    return pts3D.T

def visualize_3d_points(points_3d):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='w', marker='o', s=50, edgecolor='b')

    # Set axis limits
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    
    # Set up sliders for rotation
    ax_azim = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    azim_slider = widgets.Slider(ax_azim, 'Azimuth', -180, 180, valinit=0)
    ax_elev = fig.add_axes([0.25, 0.0, 0.65, 0.03])
    elev_slider = widgets.Slider(ax_elev, 'Elevation', -90, 90, valinit=0)

    def update_view(val):
        azim = azim_slider.val
        elev = elev_slider.val
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    azim_slider.on_changed(update_view)
    elev_slider.on_changed(update_view)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points')
    plt.show()

def fit_plane_ransac(points, distance_threshold=0.5, num_iterations=1000):
    # Convert points to Open3D point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    
    # Fit plane using RANSAC
    plane_model, inliers = cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)
    
    return plane_model, inliers

def plot_planes(points, plane_models, inliers_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', s=1)

    colors = ['r', 'g', 'y', 'c', 'm']
    for i, (plane, inliers) in enumerate(zip(plane_models, inliers_list)):
        # Extract plane parameters
        a, b, c, d = plane
        # Normal vector
        normal = np.array([a, b, c])
        # Center point
        center = points[inliers].mean(axis=0)
        ax.scatter(points[inliers][:, 0], points[inliers][:, 1], points[inliers][:, 2], color=colors[i % len(colors)], s=2)
        ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=0.5, color=colors[i % len(colors)])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Detected Planes and Normals')
    plt.show()

def project_points(points, K, R, t):
    proj_points = K @ (R @ points.T + t)
    proj_points = proj_points[:2] / proj_points[2]
    return proj_points.T

def draw_planes_on_images(img, points, color):
    for point in points:
        img = cv2.circle(img, (int(point[0]), int(point[1])), 3, color, -1)
    return img

def visualize_3d_planes(points_3d, plane_models, inliers_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='b', s=1)
    
    colors = ['r', 'g', 'y']
    for i, (plane, inliers) in enumerate(zip(plane_models, inliers_list)):
        plane_points = points_3d[inliers]
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], color=colors[i % len(colors)], s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Detected Planes in 3D Space')
    plt.show()

def visualize_planes_on_images(img1, img2, kp1, kp2, inliers_list, title='Planes Visualization'):
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for idx, inliers in enumerate(inliers_list):
        color = colors[idx % len(colors)]
        for i in inliers:
            pt1 = kp1[i].pt
            pt2 = kp2[i].pt
            cv2.circle(img1_copy, (int(pt1[0]), int(pt1[1])), 5, color, -1)
            cv2.circle(img2_copy, (int(pt2[0]), int(pt2[1])), 5, color, -1)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.title(f'{title} - Image 1')
    plt.imshow(cv2.cvtColor(img1_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(122)
    plt.title(f'{title} - Image 2')
    plt.imshow(cv2.cvtColor(img2_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

def main():
    # Set the data directory
    data_dir = 'data/example_3'
    
    # Load the input data
    img1, img2, K = load_data(data_dir)
    
    # Step 1: Find interest points
    kp1, des1 = find_interest_points(img1)
    kp2, des2 = find_interest_points(img2)
    
    # Visualize the interest points
    visualize_interest_points(img1, img2, kp1, kp2)
    
    # Step 2: Match interest points (without filtering)
    matches = match_interest_points(des1, des2, ratio_test=0.9, num_matches=100)  # Now flexible to change

    # Visualize the matched interest points (keypoints only, before filtering)
    visualize_matched_points(img1, img2, kp1, kp2, matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points (Before Filtering)')
    
    # Step 3: Filter the Matches
    inlier_matches, E, F = filter_matches_based_on_E_and_F(kp1, kp2, matches, K)
    
    # Print E and F
    print("Fundamental matrix:")
    print(F)
    
    # Print "Essential matrix:"
    print(E)
    
    # Visualize the matched interest points (keypoints + lines, after filtering)
    visualize_matches(img1, img2, kp1, kp2, inlier_matches, color1=(0, 255, 0), color2=(0, 0, 255), title='Matched Interest Points with Lines (After Filtering)')
    
    # Visualize the epipolar lines
    visualize_epipolar_lines(img1, img2, kp1, kp2, inlier_matches, F, title='Epipolar Lines')

    # Step 4: 3D reconstruction
    P1, P2_list = decompose_essential_matrix(E, K)

    points_3d = triangulate_points(kp1, kp2, P1, P2_list[0], inlier_matches)
    print(f"Number of 3D points: {points_3d.shape[0]}")
    visualize_3d_points(points_3d)

    num_planes = 2
    distance_threshold = 0.01
    num_iterations = 2000

    plane_models = []
    inliers_list = []

    remaining_points = points_3d.copy()

    for _ in range(num_planes):
        plane_model, inliers = fit_plane_ransac(remaining_points, distance_threshold, num_iterations)
        plane_models.append(plane_model)
        inliers_list.append(inliers)
        remaining_points = np.delete(remaining_points, inliers, axis=0)

    # Visualize the 3D points and the detected planes
    plot_planes(points_3d, plane_models, inliers_list)

    visualize_planes_on_images(img1, img2, kp1, kp2, inliers_list)

if __name__ == '__main__':
    main()

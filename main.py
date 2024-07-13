#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interactive, interact, interactive_output
import matplotlib.widgets as widgets

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
    plt.axis('off')
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
    h, w = shape[:2]
    if abs(b) > 1e-5:
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
    else:
        x0, y0 = int(-c / a), 0
        x1, y1 = int(-(c + b * h) / a), h
    x0, y0 = np.clip([x0, y0], 0, [w-1, h-1])
    x1, y1 = np.clip([x1, y1], 0, [w-1, h-1])
    return int(x0), int(y0), int(x1), int(y1)


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

def get_camera_projection_matrices(E, K):
    # Decompose the Essential matrix to get rotation and translation
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Construct the camera projection matrices
    P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  # P1 = K * [I | 0]
    P1 = K @ P1
    P2_list = [
        K @ np.hstack((R1, t)),  # P2 = K * [R1 | t]
        K @ np.hstack((R2, t))   # P2 = K * [R2 | t]
    ]

    return P1, P2_list

def triangulate_points(kp1, kp2, matches, P1, P2_list):
    best_pts3D = None
    best_num_points = 0

    for P2 in P2_list:
        pts3D = _triangulate_points(kp1, kp2, matches, P1, P2)
        num_points = pts3D.shape[0]
        if num_points > best_num_points:
            best_pts3D = pts3D
            best_num_points = num_points

    return best_pts3D

def _triangulate_points(kp1, kp2, matches, P1, P2):
    # Extract the matched 2D points
    pts1_hom = np.float32([kp1[m.queryIdx].pt + (1,) for m in matches]).T
    pts2_hom = np.float32([kp2[m.trainIdx].pt + (1,) for m in matches]).T

    # Triangulate the 3D points
    pts4D = cv2.triangulatePoints(P1, P2, pts1_hom[:2], pts2_hom[:2])

    # Convert homogeneous 3D points to Euclidean coordinates
    pts3D = (pts4D[:3] / pts4D[3]).T

    return pts3D

def visualize_3d_points(pts3D):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], c='w', marker='o', s=50, edgecolor='b')

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

def find_planes(pts3D, distance_threshold=0.1, min_inliers=100):
    plane_normals = []
    plane_offsets = []
    plane_colors = []

    remaining_points = pts3D.copy()
    color_index = 0

    while remaining_points.shape[0] > min_inliers:
        # Fit a plane using RANSAC
        best_normal = None
        best_offset = None
        best_inliers = 0

        for _ in range(100):
            # Randomly select 3 points to define a plane
            sample_idx = np.random.choice(remaining_points.shape[0], size=3, replace=False)
            sample_points = remaining_points[sample_idx]

            # Compute the normal vector
            normal = np.cross(sample_points[1] - sample_points[0], sample_points[2] - sample_points[0])
            normal /= np.linalg.norm(normal)

            # Compute the offset
            offset = -np.dot(normal, sample_points[0])

            # Count the number of inliers
            distances = np.abs(np.dot(remaining_points, normal) + offset) / np.linalg.norm(normal)
            inliers = np.sum(distances < distance_threshold)

            if inliers > best_inliers:
                best_normal = normal
                best_offset = offset
                best_inliers = inliers

        if best_inliers < min_inliers:
            break

        # Add the plane to the list
        plane_normals.append(best_normal)
        plane_offsets.append(best_offset)
        plane_colors.extend([color_index] * best_inliers)

        # Remove the inlier points from the remaining points
        inlier_mask = np.abs(np.dot(remaining_points, best_normal) + best_offset) / np.linalg.norm(best_normal) < distance_threshold
        remaining_points = remaining_points[~inlier_mask]

        color_index += 1

    return np.array(plane_normals), np.array(plane_offsets), np.array(plane_colors)

def visualize_planes(img1, img2, pts3D, plane_normals, plane_offsets, plane_colors):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # Plot the 2D images with color-coded points
    ax2.imshow(img1)
    ax2.imshow(img2)

    # Plot the 3D point cloud with color-coded points
    if pts3D.shape[0] == plane_colors.shape[0]:
        scat = ax.scatter(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], c=plane_colors, s=50, edgecolor='k')
    else:
        print(f"Warning: The number of 3D points ({pts3D.shape[0]}) does not match the number of plane colors ({plane_colors.shape[0]}). Plotting a subset of the points.")
        unique_colors, counts = np.unique(plane_colors, return_counts=True)
        max_points = min(counts)
        for i, color in enumerate(unique_colors):
            idx = np.where(plane_colors == color)[0][:max_points]
            ax.scatter(pts3D[idx, 0], pts3D[idx, 1], pts3D[idx, 2], c=[color], s=50, edgecolor='k')

    # Plot the plane normals
    for i, (normal, offset) in enumerate(zip(plane_normals, plane_offsets)):
        x, y, z = normal
        a, b, c = 0, 0, offset
        ax.quiver(pts3D[plane_colors == i, 0].mean(),
                  pts3D[plane_colors == i, 1].mean(),
                  pts3D[plane_colors == i, 2].mean(),
                  x, y, z, length=0.5, color=plane_colors[i])
        ax2.quiver(pts3D[plane_colors == i, 0].mean(),
                   pts3D[plane_colors == i, 1].mean(),
                   x, y, z, scale=20, color=plane_colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction with Planes')

    ax2.set_title('Planes with Normals')
    ax2.axis('off')

    plt.show()

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
    matches = match_interest_points(des1, des2, ratio_test=0.75, num_matches=500)

    # Step 3: Filter the Matches
    inlier_matches, E, F = filter_matches_based_on_E_and_F(kp1, kp2, matches, K)

    # Step 4: 3D Reconstruction
    P1, P2_list = get_camera_projection_matrices(E, K)
    pts3D = triangulate_points(kp1, kp2, inlier_matches, P1, P2_list)

    # Step 5: Find Planes
    plane_normals, plane_offsets, plane_colors = find_planes(pts3D)

    # Step 6: Visualize
    visualize_epipolar_lines(img1, img2, kp1, kp2, inlier_matches, F, title='Epipolar Lines')
    visualize_planes(img1, img2, pts3D, plane_normals, plane_offsets, plane_colors)

if __name__ == '__main__':
    main()
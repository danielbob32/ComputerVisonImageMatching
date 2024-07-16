#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.linear_model import LinearRegression

def load_data(data_dir):
    img1 = cv2.imread(os.path.join(data_dir, 'I1.png'), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise ValueError("Failed to load the first image")
    img2 = cv2.imread(os.path.join(data_dir, 'I2.png'), cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        raise ValueError("Failed to load the second image")
    K_file = os.path.join(data_dir, 'K.txt')
    K = load_camera_matrix(K_file)
    return img1, img2, K

def load_camera_matrix(K_file):
    K = np.loadtxt(K_file, delimiter=',')
    return K

def find_interest_points(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def draw_keypoints(image, keypoints, color='r'):
    plt.imshow(image, cmap='gray')
    for kp in keypoints:
        plt.plot(kp.pt[0], kp.pt[1], color+'o', markersize=2)
    plt.axis('off')

def plot_keypoints_on_images(img1, keypoints1, img2, keypoints2):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    draw_keypoints(img1, keypoints1)
    plt.title('Image 1 - Keypoints')
    plt.subplot(1, 2, 2)
    draw_keypoints(img2, keypoints2)
    plt.title('Image 2 - Keypoints')
    plt.show()

def draw_matches(img1, keypoints1, img2, keypoints2, matches, num_matches=50):
    matched_img = np.concatenate((img1, img2), axis=1)
    plt.figure(figsize=(20, 10))
    plt.imshow(matched_img, cmap='gray')
    for m in matches[:num_matches]:
        pt1 = (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1]))
        pt2 = (int(keypoints2[m.trainIdx].pt[0] + img1.shape[1]), int(keypoints2[m.trainIdx].pt[1]))
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'y-', linewidth=1)
        plt.plot(pt1[0], pt1[1], 'ro', markersize=5, markerfacecolor='none', markeredgewidth=1)
        plt.plot(pt2[0], pt2[1], 'go', markersize=5, markerfacecolor='none', markeredgewidth=1)
    plt.axis('off')
    plt.title('Matches')
    plt.show()

def drawlines(img1, img2, lines1, lines2, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r1, r2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r1[2]/r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0]*c)/r1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        x0, y0 = map(int, [0, -r2[2]/r2[1]])
        x1, y1 = map(int, [c, -(r2[2] + r2[0]*c)/r2[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Inliers and Epipolar Lines in First Image')
    axes[0].axis('off')
    for pt1 in pts1:
        axes[0].plot(pt1[0], pt1[1], 'go', markersize=7, markerfacecolor='none', markeredgewidth=1)
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Inliers and Epipolar Lines in Second Image')
    axes[1].axis('off')
    for pt2 in pts2:
        axes[1].plot(pt2[0], pt2[1], 'go', markersize=7, markerfacecolor='none', markeredgewidth=1)
    plt.show()
    

# def fit_plane(p1, p2, p3):
#     # Create vectors from points
#     v1 = p2 - p1
#     v2 = p3 - p1
#     # Compute the cross product to get the normal to the plane
#     cp = np.cross(v1, v2)
#     a, b, c = cp
#     # This gives us the coefficients of the plane equation: ax + by + cz + d = 0
#     d = -np.dot(cp, p1)
#     return a, b, c, d

def distance_from_plane(point, coefficients):
    a, b, c, d = coefficients
    den = np.sqrt(a**2 + b**2 + c**2)
    if den == 0:
        return np.inf  # Return a large number if the denominator is zero
    num = abs(a * point[0] + b * point[1] + c * point[2] + d)
    return num / den

def sequential_ransac(points, num_planes=2, max_trials=200, inlier_threshold=0.0001):
    remaining_points = points.copy()
    planes = []
    norms = []
    masks = []

    for _ in range(num_planes):
        if len(remaining_points) < 4:
            break

        best_inliers = []
        best_plane = None
        best_num_inliers = 0

        for _ in range(max_trials):
            sample = remaining_points[np.random.choice(len(remaining_points), 3, replace=False)]
            plane_coeffs = fit_plane(*sample)
            distances = np.array([distance_from_plane(pt, plane_coeffs) for pt in remaining_points])
            inliers = remaining_points[distances < inlier_threshold]

            if len(inliers) > best_num_inliers:
                refined_coeffs = refine_plane(inliers)
                refined_distances = np.array([distance_from_plane(pt, refined_coeffs) for pt in remaining_points])
                refined_inliers = remaining_points[refined_distances < inlier_threshold]

                if len(refined_inliers) > best_num_inliers:
                    best_inliers = refined_inliers
                    best_plane = refined_coeffs
                    best_num_inliers = len(refined_inliers)

        if best_plane is None:
            break

        planes.append(best_inliers)
        norms.append(best_plane[:3] / np.linalg.norm(best_plane[:3]))  # Normalize the normal vector
        mask = np.array([np.any(np.all(p == best_inliers, axis=1)) for p in points])
        masks.append(mask)

        remaining_points = remaining_points[~np.isin(remaining_points, best_inliers).all(axis=1)]

    return planes, masks, norms

def fit_plane(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = -np.dot(cp, p1)
    return a, b, c, d

# def distance_from_plane(point, coefficients):
#     a, b, c, d = coefficients
#     numerator = abs(a * point[0] + b * point[1] + c * point[2] + d)
#     denominator = np.sqrt(a**2 + b**2 + c**2)
#     return numerator / denominator

def refine_plane(points):
    # Use least squares to refine the plane equation
    X = points[:, :2]
    y = points[:, 2]
    reg = LinearRegression().fit(X, y)
    a, b = reg.coef_
    c = -1  # We assume the plane equation is of the form z = ax + by + d
    d = reg.intercept_
    return a, b, c, d


def main():
    data_dir = 'data/reference'
    num_matches = 70
    img1, img2, K = load_data(data_dir)
    if img1.shape[0] != img2.shape[0]:
        print(f"Resizing img2 from {img2.shape} to match img1 {img1.shape}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None)
    img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None)
    plot_keypoints_on_images(img1, keypoints1, img2, keypoints2)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches = good
    draw_matches(img1, keypoints1, img2, keypoints2, matches, num_matches)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K)
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    matches_inliers = [matches[i] for i in range(len(matches)) if mask_E.flatten()[i] == 1]
    print("Matrix (E):\n", E)
    print("\nMatrix (F):\n", F)
    draw_matches(img1, keypoints1, img2, keypoints2, matches_inliers, num_matches)
    pts1 = pts1[mask_E.ravel() == 1]
    pts2 = pts2[mask_E.ravel() == 1]
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    drawlines(img1, img2, lines1, lines2, pts1[:num_matches], pts2[:num_matches])
    
    pts1_undistorted = cv2.undistortPoints(pts1, K, None)
    pts2_undistorted = cv2.undistortPoints(pts2, K, None)

    points, R, t, mask = cv2.recoverPose(E, pts1_undistorted, pts2_undistorted, K)
    P1 = np.hstack((K, np.zeros((3, 1))))
    P2 = np.dot(K, np.hstack((R, t)))
    
    print("Camera Matrix P1:\n", P1)
    print("\nCamera Matrix P2:\n", P2)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_undistorted.reshape(-1, 2).T, pts2_undistorted.reshape(-1, 2).T)
    print("example of a point in 4d:", points_4d[:, 0])
    points_3d = points_4d[:3] / points_4d[3]

    print("Shape of points_4d:", points_4d.shape)
    print("Shape of points_3d:", points_3d.shape)
    print("Number of 3D points reconstructed:", points_3d.shape[1])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'3D Reconstruction Cloud ({points_3d.shape[1]} points)')
    plt.show()
    
    num_planes = 2
   
    planes, masks, normals = sequential_ransac(points_3d.T, num_planes)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Image 1 with Colored Planes')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image 2 with Colored Planes')

    colors = ['r', 'g', 'b']
    for i, inliers in enumerate(planes):
        mask = masks[i]
        projected_points_img1 = pts1[mask]
        projected_points_img2 = pts2[mask]
        for pt in projected_points_img1:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img1.shape[0] and 0 <= pt[0] < img1.shape[1]:
                axes[0].scatter(pt[0], pt[1], color=colors[i], s=5)

        for pt in projected_points_img2:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img2.shape[0] and 0 <= pt[0] < img2.shape[1]:
                axes[1].scatter(pt[0], pt[1], color=colors[i], s=5)

    for ax in axes:
        ax.axis('off')
    plt.show()
    
    #draw rectangle of the assumed planes
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Image 1 with Planes')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image 2 with Planes')
    
    for i, inliers in enumerate(planes):
        mask = masks[i]
        projected_points_img1 = pts1[mask]
        projected_points_img2 = pts2[mask]
        min_x, min_y = np.min(projected_points_img1, axis=0)
        max_x, max_y = np.max(projected_points_img1, axis=0)
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        min_x, min_y = np.min(projected_points_img2, axis=0)
        max_x, max_y = np.max(projected_points_img2, axis=0)
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='b', facecolor='none')
        axes[1].add_patch(rect)
        
    for ax in axes:
        ax.axis('off')
    plt.show()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assuming 'planes' contains the points for each plane detected
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']  # Extend colors as needed
    for plane_index, plane_points in enumerate(planes):
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], c=colors[plane_index % len(colors)], label=f'Plane {plane_index + 1}')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Reconstruction Cloud from Matches (Clustered by Planes)')
    plt.legend()
    plt.show()

    # New plane normal visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('Image 1 with Plane Normals')
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Image 2 with Plane Normals')

    scale = 50
    colors = ['blue', 'cyan']
    for i, (inliers, mask, normal) in enumerate(zip(planes, masks, normals)):
        projected_points_img1 = pts1[mask]
        projected_points_img2 = pts2[mask]
        for pt in projected_points_img1:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img1.shape[0] and 0 <= pt[0] < img1.shape[1]:
                ax1.plot([pt[0], pt[0] + scale * normal[0]], 
                         [pt[1], pt[1] + scale * normal[1]], 
                         color=colors[i], linewidth=1, solid_capstyle='round')
        for pt in projected_points_img2:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img2.shape[0] and 0 <= pt[0] < img2.shape[1]:
                ax2.plot([pt[0], pt[0] + scale * normal[0]], 
                         [pt[1], pt[1] + scale * normal[1]], 
                         color=colors[i], linewidth=1, solid_capstyle='round')
    for ax in (ax1, ax2):
        ax.axis('off')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()

# %%

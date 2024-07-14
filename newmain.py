#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
import os

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

def sequential_ransac(points, num_planes=2, max_trials=2000, residual_threshold=0.0001):
    planes = []
    remaining_points = points.copy()
    all_masks = []  # This will store the full-sized masks
    norms = []
    plane_count = 0
    total_points = points.shape[0]
    
    for i in range(num_planes):
        if len(remaining_points) < 3:
            print(f"Not enough points to form a plane. Remaining points: {len(remaining_points)}")
            break

        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(remaining_points[:, :2])
        y = remaining_points[:, 2]
        ransac = RANSACRegressor(min_samples=3, residual_threshold=residual_threshold, max_trials=max_trials)
        ransac.fit(X_poly, y)

        inlier_mask_small = ransac.inlier_mask_
        outlier_mask_small = np.logical_not(inlier_mask_small)
        
        # Convert small mask to full-sized mask
        full_mask = np.zeros(total_points, dtype=bool)
        full_mask[np.where(all_masks[-1] if all_masks else np.ones(total_points, dtype=bool))[0][:len(inlier_mask_small)]] = inlier_mask_small
        
        inliers = points[full_mask]
        planes.append(inliers)

        # Update remaining points
        remaining_points = remaining_points[outlier_mask_small]

        # Debug outputs
        print(f"Plane {i + 1}:")
        print(f"  Normal: {ransac.estimator_.coef_[1:]}, Intercept: {-ransac.estimator_.intercept_}")
        print(f"  Points on plane: {len(inliers)}")
        print(f"  Remaining points: {len(remaining_points)}")

        # Store the full mask and normals
        all_masks.append(full_mask)
        norms.append([ransac.estimator_.coef_[1], ransac.estimator_.coef_[2], -ransac.estimator_.intercept_])

        plane_count += 1

    print(f"Total planes detected: {plane_count}")
    return planes, all_masks, norms


def main():
    data_dir = 'data/reference'
    num_matches = 300
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
        if m.distance < 0.9 * n.distance:
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
    points_3d = points_4d[:3] / points_4d[3]
    clustering = DBSCAN(eps=0.1, min_samples=5).fit(points_3d.T)
    labels = clustering.labels_
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'cyan', 'red', 'green', 'magenta', 'yellow']
    for i in range(max(labels) + 1):
        cluster_points = points_3d.T[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[i % len(colors)])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Reconstruction Cloud from Matches (Clustered)')
    plt.show()
    num_planes = 2
    planes, masks, normals = sequential_ransac(points_3d.T, num_planes)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Image 1 with Colored Planes')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image 2 with Colored Planes')
    colors = ['blue', 'cyan']
    for i, mask in enumerate(masks):
        projected_points_img1 = pts1[mask]
        projected_points_img2 = pts2[mask]
        for pt in projected_points_img1:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img1.shape[0] and 0 <= pt[0] < img1.shape[1]:
                axes[0].scatter(pt[0], pt[1], color=colors[i], s=10, alpha=0.7)
        for pt in projected_points_img2:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img2.shape[0] and 0 <= pt[0] < img2.shape[1]:
                axes[1].scatter(pt[0], pt[1], color=colors[i], s=10, alpha=0.7)
    for ax in axes:
        ax.axis('off')
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Image 1 with Colored Planes and Normals')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image 2 with Colored Planes and Normals')
    scale = 20
    print(normals)
    colors = ['blue', 'cyan']
    for i, (inliers, mask, normal) in enumerate(zip(planes, masks, normals)):
        projected_points_img1 = pts1[mask]
        projected_points_img2 = pts2[mask]
        for pt in projected_points_img1:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img1.shape[0] and 0 <= pt[0] < img1.shape[1]:
                axes[0].scatter(pt[0], pt[1], color=colors[i], s=10, alpha=0.7)
                end_point = (pt[0] + scale * normal[0], pt[1] + scale * normal[1])
                axes[0].plot([pt[0], end_point[0]], [pt[1], end_point[1]], color=colors[i], linewidth=2)
        for pt in projected_points_img2:
            pt = np.round(pt).astype(int)
            if 0 <= pt[1] < img2.shape[0] and 0 <= pt[0] < img2.shape[1]:
                axes[1].scatter(pt[0], pt[1], color=colors[i], s=10, alpha=0.7)
                end_point = (pt[0] + scale * normal[0], pt[1] + scale * normal[1])
                axes[1].plot([pt[0], end_point[0]], [pt[1], end_point[1]], color=colors[i], linewidth=2)
    for ax in axes:
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

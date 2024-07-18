#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_data(data_dir):
    # Load the first image
    img1 = cv2.imread(os.path.join(data_dir, 'I1.png'), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise ValueError("Failed to load the first image")
    # Load the second image
    img2 = cv2.imread(os.path.join(data_dir, 'I2.png'), cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        raise ValueError("Failed to load the second image")
    K_file = os.path.join(data_dir, 'K.txt')
    K = load_camera_matrix(K_file)
    return img1, img2, K

def load_camera_matrix(K_file):
    # Load the camera matrix from a text file
    K = np.loadtxt(K_file, delimiter=',')
    return K

def find_interest_points(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def find_matches(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]

def find_F_and_E_inlinersByE(kp1,kp2,K,matches):
    # Get points from the keypoint matches
    mp1 = np.array([kp1[m.queryIdx].pt for m in matches])
    mp2 = np.array([kp2[m.trainIdx].pt for m in matches])
    # Find essential and fundamental matrices
    E, E_mask = cv2.findEssentialMat(mp1, mp2, K)
    F, F_mask = cv2.findFundamentalMat(mp1, mp2,cv2.RANSAC)
    # Filter matches based on the essential mask
    match_inlieers = []
    for i,match1 in enumerate(matches):
        if E_mask.flatten()[i] == 1:
          match_inlieers.append(match1)
    return mp1, mp2, E, E_mask, F, F_mask, match_inlieers


def find_planes(points, planes_amount, Ransac_iterations, thereshold):
    unfitted_points = points.copy()
    normals = []
    planes = []

    for i in range(planes_amount):
        if len(unfitted_points) < 3:
            break

        good_inliers_mask = []
        hyp = None

        for _ in range(Ransac_iterations):
            idx = np.random.permutation(len(unfitted_points))[:3]
            p1, p2, p3 = unfitted_points[idx[0]], unfitted_points[idx[1]], unfitted_points[idx[2]]
            norm = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(norm) == 0:
                continue
            norm = norm / np.linalg.norm(norm)
            a, b, c = norm
            d = -np.dot(norm, p1)
            inliers_mask = []
            for i, pt in enumerate(unfitted_points):
                x, y, z = pt
                distance = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                inliers_mask.append(distance < thereshold)
            if sum(inliers_mask) > sum(good_inliers_mask):
                hyp = (a, b, c, d)
                good_inliers_mask = inliers_mask

        inliers_mask = np.array(good_inliers_mask).astype(bool)

        inliers = unfitted_points[inliers_mask]
        outliers = unfitted_points[~inliers_mask]
        unfitted_points = outliers
        planes.append(inliers)
        normals.append(hyp)
    return normals, planes

def compute_points_cloud(mp1,mp2, K ,E):
    un_mp1 = cv2.undistortPoints(mp1, K, None)
    un_mp2 = cv2.undistortPoints(mp2, K, None)
    points, R, t , _ = cv2.recoverPose(E, mp1, mp2, K)
    P1 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]).astype(np.float32)
    P2 = np.hstack((R, t))
    
    # # plot 3D points cloud
    # fig = plt.figure(figsize=(10, 8))
    # pl = fig.add_subplot(111, projection='3d')
    # pl.scatter(threeD_points[0], threeD_points[1], threeD_points[2], color='b', marker='o')
    # pl.set_xlabel('X')
    # pl.set_ylabel('Y')
    # pl.set_zlabel('Z')
    # plt.show()
    
    points_homo = cv2.triangulatePoints(P1, P2, un_mp1, un_mp2)
    threeD_points = points_homo[:3] / points_homo[3]
    return threeD_points,P1,P2,R ,t



################################################ visuals ###################################################################################

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    points = []
    for i in np.arange(0, dist, gap):
        z = i / dist
        x = int((pt1[0] * (1 - z) + pt2[0] * z) + 0.5)
        y = int((pt1[1] * (1 - z) + pt2[1] * z) + 0.5)
        points.append((x, y))
    for point in points:
        cv2.circle(img, point, thickness, color, -1)  
        
def plot_keypoints(img1, kp1,img2, kp2):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    for kp in kp1:
        plt.plot(kp.pt[0], kp.pt[1], 'ro', markersize=1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    for kp in kp2:
        plt.plot(kp.pt[0], kp.pt[1], 'ro', markersize=1)
    plt.show()
    
def plot_matches(img1, kp1, img2, kp2, matches):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    for match in matches:
        kp = kp1[match.queryIdx]
        plt.plot(kp.pt[0], kp.pt[1], 'ro', markersize=1)
    plt.subplot(1, 2, 2)    
    plt.imshow(img2, cmap='gray')
    for match in matches:
        kp = kp2[match.trainIdx]
        plt.plot(kp.pt[0], kp.pt[1], 'ro', markersize=1)
    plt.show()
    
def plot_matches_with_lines(img1, kp1, img2, kp2, matches,amount):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    height = max(h1, h2)
    width = w1 + w2
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    output_image[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    output_image[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for match in matches[:amount]:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt
        
        cv2.circle(output_image, (int(x1), int(y1)), 7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(output_image, (int(x2 + w1), int(y2)), 7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        
        draw_dotted_line(output_image, (int(x1), int(y1)), (int(x2 + w1), int(y2)), (0, 255, 255), thickness=1, gap=5)

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_epi_lines(img1, img2, mps1, mps2, F, E_mask, amount):
    # Apply mask to get inliers
    mps1 = mps1[E_mask.flatten() == 1]
    mps2 = mps2[E_mask.flatten() == 1]
    
    # Select a random sample of inliers if the amount is less than the number of inliers
    if amount < len(mps1):
        indices = np.random.choice(len(mps1), amount, replace=False)
        mps1_ = mps1[indices]
        mps2_ = mps2[indices]
    else:
        mps1_ = mps1[:amount]
        mps2_ = mps2[:amount]
    
    # Compute epipolar lines for the selected points
    image1_lines = cv2.computeCorrespondEpilines(mps2_.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    image2_lines = cv2.computeCorrespondEpilines(mps1_.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    # Convert images to color for visualization
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw epipolar lines
    _, w1 = img1.shape
    for (line1, mp1, line2, mp2) in zip(image1_lines, mps1_, image2_lines, mps2_):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Line on first image
        x0, y0 = map(int, [0, -line1[2]/line1[1]])
        x1, y1 = map(int, [w1, -(line1[2] + line1[0]*w1)/line1[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        
        # Line on second image
        x0, y0 = map(int, [0, -line2[2]/line2[1]])
        x1, y1 = map(int, [w1, -(line2[2] + line2[0]*w1)/line2[1]])
        img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
    
    # Plotting
    _, plts = plt.subplots(1, 2, figsize=(20, 10))
    plts[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    plts[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
    for mp1, mp2 in zip(mps1_, mps2_):
        plts[0].plot(mp1[0], mp1[1], 'go', markersize=6, markerfacecolor='none', markeredgewidth=1)
        plts[1].plot(mp2[0], mp2[1], 'go', markersize=6, markerfacecolor='none', markeredgewidth=1)
    plt.show()

    return mps1, mps2


def plot_planes(normals, planes, image1, image2, P1, K, R, t):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image1, cmap='gray')
    axes[1].imshow(image2, cmap='gray')
    colors = ['blue', 'cyan', 'yellow', 'red', 'green']

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)  

    for i, (normal, plane) in enumerate(zip(normals, planes)):
        color = colors[i % len(colors)]
        plane_points = np.array(plane, dtype=np.float32).reshape(-1, 1, 3)
        dist_coeffs = np.zeros(5)  

        # Project points onto the image planes using the camera parameters
        projected_points1, _ = cv2.projectPoints(plane_points, np.zeros(3), np.zeros(3), K, dist_coeffs)
        projected_points2, _ = cv2.projectPoints(plane_points, rvec, t, K, dist_coeffs) 

        # Draw points on the images
        for pt1, pt2 in zip(projected_points1, projected_points2):
            pt1 = pt1.ravel().astype(int)
            pt2 = pt2.ravel().astype(int)
            if 0 <= pt1[0] < image1.shape[1] and 0 <= pt1[1] < image1.shape[0]:
                axes[0].scatter(pt1[0], pt1[1], color=color, s=5)
            if 0 <= pt2[0] < image2.shape[1] and 0 <= pt2[1] < image2.shape[0]:
                axes[1].scatter(pt2[0], pt2[1], color=color, s=5)

    plt.tight_layout()
    plt.show()


def plot_normals(normals, planes, image1, image2, P1, K, R, t):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image1, cmap='gray')
    axes[1].imshow(image2, cmap='gray')
    
    colors = ['blue', 'cyan', 'red', 'green', 'magenta', 'yellow']  # Colors for different planes

    for idx, (normal, plane) in enumerate(zip(normals, planes)):
        color = colors[idx % len(colors)]  # Cycle through colors

        # Extract the directional components of the normal vector (ignoring d)
        normal_vector = np.array(normal[:3]).reshape(1, 3)

        # Convert plane to a suitable shape for projection
        plane_points = np.array(plane, dtype=np.float32).reshape(-1, 1, 3)

        # Calculate endpoints for the normals visualization
        scaled_normals = 1* normal_vector  # Scale for visualization
        end_points = plane_points + scaled_normals

        # Project points onto the image planes
        img_points1, _ = cv2.projectPoints(plane_points, np.zeros(3), np.zeros(3), K, None)
        img_points2, _ = cv2.projectPoints(plane_points, R, t, K, None)
        end_img_points1, _ = cv2.projectPoints(end_points, np.zeros(3), np.zeros(3), K, None)
        end_img_points2, _ = cv2.projectPoints(end_points, R, t, K, None)

        # Draw lines on the images
        for (pt1, end_pt1), (pt2, end_pt2) in zip(zip(img_points1, end_img_points1), zip(img_points2, end_img_points2)):
            pt1 = tuple(pt1.ravel().astype(int))
            end_pt1 = tuple(end_pt1.ravel().astype(int))
            pt2 = tuple(pt2.ravel().astype(int))
            end_pt2 = tuple(end_pt2.ravel().astype(int))

            # Check both point and end-point are within image bounds before drawing
            if 0 <= pt1[0] < image1.shape[1] and 0 <= pt1[1] < image1.shape[0] and \
               0 <= end_pt1[0] < image1.shape[1] and 0 <= end_pt1[1] < image1.shape[0]:
                axes[0].plot([pt1[0], end_pt1[0]], [pt1[1], end_pt1[1]], color=color, linewidth=0.5)

            if 0 <= pt2[0] < image2.shape[1] and 0 <= pt2[1] < image2.shape[0] and \
               0 <= end_pt2[0] < image2.shape[1] and 0 <= end_pt2[1] < image2.shape[0]:
                axes[1].plot([pt2[0], end_pt2[0]], [pt2[1], end_pt2[1]], color=color, linewidth=0.5)

    plt.tight_layout()
    plt.show()
######################################################################################################################### 

def main():
    
    data_dir = 'data/example_3'
    amount = 70
    planes_amount = 2
    Ransac_iterations = 1000
    threshold = 0.5

    
    img1, img2, K = load_data(data_dir)
    
    if img1.shape[0] != img2.shape[0]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    kp1, des1 = find_interest_points(img1)
    kp2, des2 = find_interest_points(img2)
    
    plot_keypoints(img1, kp1,img2, kp2)
    
    matches = find_matches(des1, des2)
    
    plot_matches(img1, kp1, img2, kp2, matches)
    
    plot_matches_with_lines(img1, kp1, img2, kp2, matches,amount)
    
    mp1, mp2, E, E_mask, F, F_mask, match_inliners = find_F_and_E_inlinersByE(kp1,kp2,K,matches)
    
    print("\n\nMatrix (E):\n", E)
    print("\nMatrix (F):\n\n", F)
    
    plot_matches_with_lines(img1, kp1, img2, kp2, match_inliners,amount)
    
    mp1,mp2 = plot_epi_lines(img1, img2, mp1, mp2, F,E_mask,amount)
    
    threeD_points,P1,P2,R,t =compute_points_cloud(mp1,mp2, K ,E)
    
    print("Camera Matrix P1:\n", P1)
    print("\nCamera Matrix P2:\n", P2)
    
    normals, planes = find_planes(threeD_points.T, planes_amount,Ransac_iterations,threshold)
    
    plot_planes(normals, planes,img1,img2,P1,K,R,t)
    
    plot_normals(normals, planes,img1,img2,P1,K,R,t)
    
if __name__ == '__main__':
    main()
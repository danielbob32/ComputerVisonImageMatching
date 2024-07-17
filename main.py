#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    
def find_matches(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_F_and_E_inlinersByE(kp1,kp2,K,matches):
    mp1 = np.array([kp1[m.queryIdx].pt for m in matches])
    mp2 = np.array([kp2[m.trainIdx].pt for m in matches])
    E, E_mask = cv2.findEssentialMat(mp1, mp2, K)
    F, F_mask = cv2.findFundamentalMat(mp1, mp2,cv2.RANSAC)
    match_inliners =  [matches[i] for i in range(len(matches)) if E_mask.flatten()[i] == 1]
    return mp1, mp2, E, E_mask, F, F_mask, match_inliners

def find_planes(points, planes_amount, Ransac_iterations, thereshold):
    unfitted_points = points.copy()
    normals = []
    planes = []
    
    for i in range(planes_amount):
        if len(unfitted_points) < 3:
            break
        hyp, inliers, outliers = fit_planes(unfitted_points, Ransac_iterations, thereshold)
        unfitted_points = outliers
        planes.append(inliers)
        normals.append(hyp)
    return normals, planes

def fit_planes(unfitted_points, Ransac_iterations , threshold):
    good_inliers = []
    good_outliers = []
    model = None
    
    for _ in range(Ransac_iterations):
        p1, p2, p3 = unfitted_points[np.random.choice(unfitted_points.shape[0], 3, replace=False)]
        norm = np.cross(p2 - p1, p3 - p1)
        if np.isclose(np.linalg.norm(norm), 0):
            continue
        norm = norm / np.linalg.norm(norm)
        a, b, c = norm
        d = -np.dot(norm, p1)
        inliers = []
        outliers = []
        for i, pt in enumerate(unfitted_points):
            x, y, z = pt
            distance = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
            if distance < threshold:
                inliers.append(pt)
            else:
                outliers.append(pt)
                
        if len(inliers) > len(good_inliers):
            model = (a, b, c, d)
            good_inliers = inliers
            good_outliers = outliers
           
    return model, good_inliers.copy(), np.array(good_outliers.copy())

def compute_points_cloud(mp1,mp2, K ,E):
    un_mp1 = cv2.undistortPoints(mp1, K, None)
    un_mp2 = cv2.undistortPoints(mp2, K, None)
    points, R, t , mask = cv2.recoverPose(E, mp1, mp2, K)
    P1 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]], dtype=np.float32)
    P2 = np.hstack((R, t))
    
    points_homo = cv2.triangulatePoints(P1, P2, un_mp1, un_mp2)
    threeD_points = points_homo[:3] / points_homo[3]
    
    # # plot 3D points cloud
    # fig = plt.figure(figsize=(10, 8))
    # pl = fig.add_subplot(111, projection='3d')
    # pl.scatter(threeD_points[0], threeD_points[1], threeD_points[2], color='b', marker='o')
    # pl.set_xlabel('X')
    # pl.set_ylabel('Y')
    # pl.set_zlabel('Z')
    # plt.show()
    
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
    
def plot_epi_lines(img1, img2, mps1, mps2, F, E_mask,amount):
    
    mps1 = mps1[E_mask.flatten() == 1]
    mps2 = mps2[E_mask.flatten() == 1]
    mps1_ = mps1[:amount]
    mps2_ = mps2[:amount]
    image1_lines = cv2.computeCorrespondEpilines(mps2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    image2_lines = cv2.computeCorrespondEpilines(mps1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    _, w1 = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for z1,mp1,z2,mp2 in zip(image1_lines,mps1_, image2_lines,mps2_):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -z1[2]/z1[1]])
        x1, y1 = map(int, [w1, -(z1[2] + z1[0]*w1)/z1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        x0, y0 = map(int, [0, -z2[2]/z2[1]])
        x1, y1 = map(int, [w1, -(z2[2] + z2[0]*w1)/z2[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
    _, plts = plt.subplots(1, 2, figsize=(20, 10))
    plts[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    for mp1 in mps1_:
        plts[0].plot(mp1[0], mp1[1], 'go', markersize=6, markerfacecolor='none', markeredgewidth=1)
    plts[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    for mp2 in mps2_:
        plts[1].plot(mp2[0], mp2[1], 'go', markersize=6, markerfacecolor='none', markeredgewidth=1)
    plt.show()
    
    return mps1,mps2

def plot_planes(normals, planes,image1,image2,P1,K,R,t):
    fig, pl = plt.subplots(1, 2, figsize=(20, 10))
    pl[0].imshow(image1, cmap='gray')
    pl[1].imshow(image2, cmap='gray')

    colors = ['blue', 'cyan']
    for i, inliers in enumerate(planes):
        projected_points1 = cv2.projectPoints(np.array(inliers), cv2.Rodrigues(P1[:,:3])[0], P1[:, 3:], K, None)[0].squeeze()
        projected_points2= cv2.projectPoints(np.array(inliers), R, t, K, None)[0].squeeze()
        for point in projected_points1:
            point = np.round(point).astype(int)
            if 0 <= point[1] < image1.shape[0] and 0 <= point[0] < image1.shape[1]:
                pl[0].scatter(point[0], point[1], color=colors[i], s=5)

        for point in projected_points2:
            point = np.round(point).astype(int)
            if 0 <= point[1] < image2.shape[0] and 0 <= point[0] < image2.shape[1]:
                pl[1].scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()

def plot_normals(normals, planes,image1,image2,P1,K,R,t):
    fig, pl = plt.subplots(1, 2, figsize=(20, 10))
    pl[0].imshow(image1, cmap='gray')
    pl[1].imshow(image2, cmap='gray')
    scaling = 1
    colors = ['blue', 'cyan']
    for i, (normal, plane) in enumerate(zip(normals, planes)):
        projected_points1 = cv2.projectPoints(np.array(plane), cv2.Rodrigues(P1[:, :3])[0], P1[:, 3:], K, None)[0].squeeze()
        projected_points2 = cv2.projectPoints(np.array(plane), R, t, K, None)[0].squeeze()
        threeD_points = plane - scaling * np.array(normal[:3])
        projected_points1_ = cv2.projectPoints(threeD_points, cv2.Rodrigues(P1[:, :3])[0], P1[:, 3:], K, None)[0].squeeze()
        projected_points2_ = cv2.projectPoints(threeD_points, R, t, K, None)[0].squeeze()

        for point, point_ in zip(projected_points1, projected_points1_):
            point = np.round(point).astype(int)
            point_ = np.round(point_).astype(int)
            # Check both point and point_ are within margins
            if all(0 < p < lim for p, lim in zip(point, image1.shape[::-1])) and \
               all(0 < p < lim for p, lim in zip(point_, image1.shape[::-1])):
                pl[0].plot([point[0], point_[0]], [point[1], point_[1]], color=colors[i], linewidth=0.5)
            
        for point, point_ in zip(projected_points2, projected_points2_):
            point = np.round(point).astype(int)
            point_ = np.round(point_).astype(int)
            # Check both point and point_ are within margins
            if all(0 < p < lim for p, lim in zip(point, image2.shape[::-1])) and \
               all(0 < p < lim for p, lim in zip(point_, image2.shape[::-1])):
                pl[1].plot([point[0], point_[0]], [point[1], point_[1]], color=colors[i], linewidth=0.5)

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

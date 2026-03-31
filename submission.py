"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # Step 1: Normalize points
    T = np.array([
    [1/M, 0,   0],
    [0,   1/M, 0],
    [0,   0,   1]
    ])

    # Convert to homogeneous
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    # Apply transform
    pts1_norm = (T @ pts1_h.T).T
    pts2_norm = (T @ pts2_h.T).T
    
    # Step 2: Construct matrix A
    N = pts1.shape[0]
    A = []

    for i in range(N):
        x1, y1 = pts1_norm[i][:2]
        x2, y2 = pts2_norm[i][:2]

        A.append([
            x2*x1, x2*y1, x2,
            y2*x1, y2*y1, y2,
            x1,    y1,    1
        ])

    A = np.array(A)

    # Step 3: Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3,3)

    # Step 4: Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Step 5: Un-normalize
    T = np.array([
        [1/M, 0,   0],
        [0,   1/M, 0],
        [0,   0,   1]
    ])

    F = T.T @ F @ T
    if abs(F[2,2]) > 1e-8:
        F = F / F[2,2]
    return F



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""



def epipolar_correspondences(im1, im2, F, pts1):

    window_size = 5
    half = window_size // 2

    pts2 = []

    # Convert to grayscale (important)
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    h, w = im2_gray.shape

    for (x, y) in pts1:

        x, y = int(x), int(y)

        # Step 1: compute epipolar line
        line = F @ np.array([x, y, 1])
        a, b, c = line

        best_error = float('inf')
        best_point = (0, 0)

        # Step 2: search along epipolar line
        for x2 in range(w):

            if abs(b) < 1e-6:
                continue

            y2 = int((-a * x2 - c) / b)

            # check bounds
            if y2 < half or y2 >= h - half or x2 < half or x2 >= w - half:
                continue

            # Step 3: extract patches
            patch1 = im1_gray[y-half:y+half+1, x-half:x+half+1]
            patch2 = im2_gray[y2-half:y2+half+1, x2-half:x2+half+1]

            # Step 4: compute error
            error = np.sum((patch1 - patch2) ** 2)

            # Step 5: update best match
            if error < best_error:
                best_error = error
                best_point = (x2, y2)

        pts2.append(best_point)

    return np.array(pts2)


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(E)

    S = [1, 1, 0]   # ideal singular values
    E = U @ np.diag(S) @ Vt
    
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass

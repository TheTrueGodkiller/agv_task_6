import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz

data = np.load("/data/some_corresp.npz")
pts1 = data['pts1']
pts2 = data['pts2']
im1=cv2.imread("/data/im1.png")
im2=cv2.imread("/data/im2.png")
#print(data.files)

# 2. Run eight_point to compute F

h, w = im1.shape[:2]
M = max(h, w)  

F = sub.eight_point(pts1, pts2, M)

print(F)

from helper import displayEpipolarF

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz

temple_data = np.load("/data/temple_coords.npz")
pts1_temple = temple_data['pts1']

# 4. Run epipolar_correspondences to get points in image 2

pts2_temple = sub.epipolar_correspondences(im1, im2, F, pts1_temple)
from helper import epipolarMatchGUI
#epipolarMatchGUI(im1, im2, F)

# 5. Compute the camera projection matrix P1

intrinsics = np.load("/data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E=sub.essential_matrix(F,K1,K2)

M1 = np.hstack((np.eye(3), np.zeros((3,1))))
P1 = K1 @ M1

# 6. Use camera2 to get 4 camera projection matrices P2
from helper import camera2
M2s = camera2(E)

'''best_P2 = None
best_pts3d = None
max_positive = 0

for i in range(4):
    M2 = M2s[:, :, i]
    P2 = K2 @ M2'''

# 7. Run triangulate using the projection matrices

''' pts3d = sub.triangulate(P1, pts1, P2, pts2)
    positive = np.sum(pts3d[:, 2] > 0)'''

# 8. Figure out the correct P2

''' if positive > max_positive:
        max_positive = positive
        best_P2 = P2
        best_pts3d = pts3d'''

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

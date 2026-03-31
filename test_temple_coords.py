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

#print(F)

from helper import displayEpipolarF

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#displayEpipolarF(im1, im2, F)

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
print(E)

M1 = np.hstack((np.eye(3), np.zeros((3,1))))
P1 = K1 @ M1


#detecting edge points in image 1 to get more correspondences for triangulation

# Convert to grayscale
gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

# Detect edges
edges = cv2.Canny(gray, 100, 200)

# Get coordinates of edge pixels
ys, xs = np.where(edges > 0)

# Combine into (x, y) format
pts1_edges = np.stack([xs, ys], axis=1)

num_points = 300  # adjust (200–500 is good)

indices = np.random.choice(len(pts1_edges), num_points, replace=False)
pts1_edges = pts1_edges[indices]
pts2_edges = sub.epipolar_correspondences(im1, im2, F, pts1_edges)

pts1_all = np.vstack((pts1_temple, pts1_edges))
pts2_all = np.vstack((pts2_temple, pts2_edges))




# 6. Use camera2 to get 4 camera projection matrices P2
from helper import camera2
M2s = camera2(E)

best_P2 = None
best_pts3d = None
max_positive = 0

for i in range(4):
    M2 = M2s[:, :, i]
    P2 = K2 @ M2

# 7. Run triangulate using the projection matrices

    pts3d = sub.triangulate(P1, pts1_all, P2, pts2_all)
    positive = np.sum(pts3d[:, 2] > 0)

# 8. Figure out the correct P2

    if positive > max_positive:
        max_positive = positive
        best_P2 = P2
        best_pts3d = pts3d
x = best_pts3d[:,0]
y = best_pts3d[:,1]
mask = (np.abs(x) < 0.5) & (np.abs(y) < 0.5)
best_pts3d = best_pts3d[mask]

# 9. Scatter plot the correct 3D points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:,0], best_pts3d[:,1], best_pts3d[:,2],s=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=10, azim=120)
ax.set_box_aspect([1,1,1])
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

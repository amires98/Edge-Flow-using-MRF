import numpy as np
import cv2
import matplotlib.pyplot as plt
from ProbabilisticTree import ProbableTree


A = np.zeros((100, 100))
B = np.zeros((100, 100))
A[30, :-40] = 1
A[20:60, 30] = 1

B[25, 5:-35] = 1
B[15:55, 35] = 1

A = A.astype(np.uint8)
B = B.astype(np.uint8)


nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(A, connectivity=8)
for i in range(1,nb_components):
    left, top, h, w, area = stats[i]
    if area <= 20:
        continue
    tree = ProbableTree(output, i, stats[i], A, B, 13, 5)
    tree.backtrack()
    print(tree.get_map())

print('done')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(A, cmap='gray')
ax[1].imshow(B, cmap='gray')
plt.gray()
plt.show()

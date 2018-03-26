import cv2
from matplotlib import pyplot as plt
import time

MIN_MATCH_COUNT = 10
# 44806beb654e6410.jpg
img1 = cv2.imread('/home/dobosevych/Public/APPS.UCU/Google-Landmark-Retrieval-Challenge/data/index/096ba6b40a226adf.jpg',0)
img2 = cv2.imread('/home/dobosevych/Public/APPS.UCU/Google-Landmark-Retrieval-Challenge/data/index/images.jpeg',0)
img3 = cv2.imread('/home/dobosevych/Public/APPS.UCU/Google-Landmark-Retrieval-Challenge/data/index/d485d9f770e40453.jpg',0)

# Initiate SIFT detector
# sift = cv2.ORB_create()
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SURF_create()

start = time.time()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
end = time.time()

print(end - start)

coeff = 0.7

start = time.time()

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des3, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < coeff * n.distance:
        good.append([m])

print(len(good))

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < coeff * n.distance:
        good.append([m])

print(len(good))
end = time.time()


end = time.time()
print(end - start)

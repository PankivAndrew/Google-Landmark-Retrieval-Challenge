# import the necessary packages
from rootsift import RootSIFT
import cv2


# load the image we are going to extract descriptors from and convert
# it to grayscale
image1 = cv2.imread("/home/chepubelja/Pictures/1-1.jpg")
image2 = cv2.imread("/home/chepubelja/Pictures/1-2.jpg")
image3 = cv2.imread("/home/chepubelja/Pictures/1-3.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # detect Difference of Gaussian keypoints in the image
# detector = cv2.xfeatures2d.SIFT_create()
# kps = detector.detect(gray)
#
# # extract normal SIFT descriptors
# extractor = cv2.xfeatures2d.SIFT_create()
# (kps, descs) = extractor.detectAndCompute(gray, None)
# print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
# #
# # extract RootSIFT descriptors
rs = RootSIFT()
kp1, des1 = rs.detectAndCompute(image1, None)
kp2, des2 = rs.detectAndCompute(image2, None)
kp3, des3 = rs.detectAndCompute(image3, None)
# print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)


img2 = cv2.drawKeypoints(image1, kp1, None, color=(0, 0, 255), flags=0)
cv2.imshow('gray_image.jpg', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
#
# coeff = 0.7
#
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < coeff * n.distance:
#         good.append([m])
#
# print(len(good))
#

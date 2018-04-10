# import the necessary packages
from rootsift import RootSIFT
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directory = "../../test_images/"
num_elements = len([name for name in os.listdir(directory)])

similarity_matrix = np.zeros((num_elements, num_elements))
# print(similarity_matrix)
rs = RootSIFT()
for i, filename_1 in enumerate(os.listdir(directory)):
    image_1 = cv2.imread(directory + filename_1)
    # extract RootSIFT descriptors
    kp1, des1 = rs.detectAndCompute(image_1, None)
    for j, filename_2 in enumerate(os.listdir(directory)):
        print(i, j)
        if filename_1 == filename_2:
            pass
        else:
            image_2 = cv2.imread(directory + filename_2)
            # extract RootSIFT descriptors
            kp2, des2 = rs.detectAndCompute(image_2, None)

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            num_of_matches = len(good)
            similarity_matrix[i][j] = num_of_matches

print(similarity_matrix)
#
#
#
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         rs = RootSIFT()
#         if i == 0:
#             temp_image = cv2.imread(directory + filename)
#             # extract RootSIFT descriptors
#             temp_kp, temp_des = rs.detectAndCompute(temp_image, None)
#         image = cv2.imread(directory + filename)
#         # extract RootSIFT descriptors
#         rs = RootSIFT()
#         kp, des = rs.detectAndCompute(image, None)
#
#         # BFMatcher with default params
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(des1, des2, k=2)
#         #
#         coeff = 0.75
#         # # Apply ratio test
#         good = []
#         for m, n in matches:
#             if m.distance < coeff * n.distance:
#                 good.append([m])
#
#         print(len(good))
#     else:
#         pass













# # load the image we are going to extract descriptors from and convert
# # it to grayscale
# image1 = cv2.imread("/home/chepubelja/Pictures/1-1.jpg")
# image2 = cv2.imread("/home/chepubelja/Pictures/1-2.jpg")
# image3 = cv2.imread("/home/chepubelja/Pictures/1-3.jpg")
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #
# # # detect Difference of Gaussian keypoints in the image
# # detector = cv2.xfeatures2d.SIFT_create()
# # kps = detector.detect(gray)
# #
# # # extract normal SIFT descriptors
# # extractor = cv2.xfeatures2d.SIFT_create()
# # (kps, descs) = extractor.detectAndCompute(gray, None)
# # print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
# # #
# # # extract RootSIFT descriptors
# rs = RootSIFT()
# kp1, des1 = rs.detectAndCompute(image1, None)
# kp2, des2 = rs.detectAndCompute(image2, None)
# kp3, des3 = rs.detectAndCompute(image3, None)
# # print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
#
#
# # img2 = cv2.drawKeypoints(image1, kp1, None, color=(0, 0, 255), flags=0)
# # cv2.imshow('gray_image.jpg', img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# #
# coeff = 0.75
# # # Apply ratio test
# good = []
# for m, n in matches:
#     if m.distance < coeff * n.distance:
#         good.append([m])
#
# print(len(good))
#
# # result = np.empty(good, good)
# # cv2.drawMatchesKnn expects list of lists as matches.
# new_image = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good, flags=2, outImg=None)
# # print(result)
# # plt.imshow(new_image), plt.show()

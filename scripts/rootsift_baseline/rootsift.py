# import the necessary packages
import numpy as np
import cv2


class RootSIFT:
    def __init__(self):
        # Initializing the SIFT feature extractor
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def detectAndCompute(self, image, kp, eps=1e-7):
        # compute SIFT descriptors
        kp, desc = self.extractor.detectAndCompute(image, kp)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kp) == 0:
            return [], None

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        desc /= (desc.sum(axis=1, keepdims=True) + eps)
        desc = np.sqrt(desc)

        # return a tuple of the keypoints and descriptors
        return kp, desc

import cv2
import numpy as np
from skimage.measure import ransac


class TranslationTransform:
    def __init__(self):
        self.dx = None
        self.dy = None

    def estimate(self, data):
        x1, y1, x2, y2 = data[0]
        self.dx = x2 - x1
        self.dy = y2 - y1
        return True

    def residuals(self, data):
        points1 = np.copy(data[:, :2])
        points2 = np.copy(data[:, 2:])
        points1 += (self.dx, self.dy)
        return [np.linalg.norm(diff, ord=2) for diff in points2 - points1]

    def get_transform_matrix(self):
        h = np.eye(3)
        h[0, 2] = self.dx
        h[1, 2] = self.dy
        return h


##############################################################
# Code taken from: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# And modified
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # x = cv2.estimateRigidTransform(points1, points2, fullAffine=False)
    model, inliers = ransac(np.concatenate((points1, points2), axis=1), TranslationTransform,
                            min_samples=1, residual_threshold=1)
    h = model.get_transform_matrix()

    # Use homography
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h, imMatches
##############################################################

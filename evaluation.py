# feature matching
import cv2
import glob
from matplotlib import image


# # BF matcher
# cách này đôi khi có sự nhầm lẫn khi có một số hình ảnh giống nhau
def BF_orb_det(img1, img2):
    """Calculate evaluation of BF matcher in ORB detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print("keypoints: {}, descriptors: {}".format(len(kp2), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))
    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # best matches in 2 images
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    correct_matches = len(matches)
    correspondences = round((len(kp1) + len(kp2) / 2))

    print(f"All matches of ORB BF: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall ORB BF: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision ORB BF: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches ORB BF: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision ORB BF: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches ORB BF: {false_matches}")

    print()
    return


# # cách này cho kết quả tốt hơn, ổn định hơn
def BF_sift_det(img1, img2):
    """Calculate evaluation of BF matcher in SIFT detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # create our SIFT detector and detect key points and descriptors
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    # Match with BF
    bf = cv2.BFMatcher()

    # knnMatch - visualize descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    correct_matches = len(matches)
    correspondences = round((len(kp1) + len(kp2) / 2))
    print(f"All matches of BF SIFT: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall BF SIFT: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision BF SIFT: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF SIFT: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision BF SIFT: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches BF SIFT: {false_matches}")
    print()

    return


def AKAZE_BF(img1, img2):
    """Calculate evaluation of BF matcher in AKAZE BF: Recall, 1-precision, precision, correct matches,
    false matches"""
    # load the image and convert it to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kp1, des1) = detector.detectAndCompute(gray1, None)
    (kp2, des2) = detector.detectAndCompute(gray2, None)

    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    # # Apply ratio test
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.35 * n.distance:
    #         good_matches.append([m])

    correct_matches = len(matches)
    correspondences = round((len(kp1) + len(kp2) / 2))
    print(f"All matches of BF AKAZE: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall BF AKAZE: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision BF AKAZE: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF AKAZE: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision BF AKAZE: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches BF AKAZE: {false_matches}")
    print()
    # cv2.drawMatchesKnn expects list of lists as matches.
    # im3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good_matches[1:20], None, flags=2)
    # cv2.imshow("AKAZE matching", im3)
    # cv2.waitKey(0)
    return


####################################################################


# # FLANN SIFT
def feat_match_FLANN_sift_visualize(img1, img2):
    """Calculate evaluation of FLANN matcher in SIFT detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    correct_matches = len(matches)
    correspondences = round((len(kp1) + len(kp2) / 2))
    print(f"All matches of FLANN SIFT: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall FLANN SIFT: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision FLANN SIFT: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches FLANN SIFT: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision FLANN SIFT: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches FLANN SIFT: {false_matches}")

    print()
    return


def feat_match_FLANN_surf_visualize(img1, img2):
    """Calculate evaluation of FLANN matcher in SURF detector: Recall, 1-precision, precision, correct matches,
    false matches """
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, 2)

    correct_matches = len(matches)
    correspondences = round((len(kp1) + len(kp2) / 2))
    print(f"All matches of SURF FLANN: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall SURF FLANN: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision SURF FLANN: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches SURF FLANN: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision SURF FLANN: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches SURF FLANN: {false_matches}")
    print()
    return


def append_image_in_folder(path):
    """Read images from specified folder"""
    images = []
    for filename in glob.glob(path + '*.ppm'):
        # load image
        img_data = image.imread(filename)
        images.append(img_data)
    return images


def run():
    """Run the program"""
    # read lst image
    img_ubc = append_image_in_folder('img/bikes/')

    # ORB BF
    BF_orb_det(img_ubc[0], img_ubc[1])
    BF_orb_det(img_ubc[0], img_ubc[2])
    BF_orb_det(img_ubc[0], img_ubc[3])
    BF_orb_det(img_ubc[0], img_ubc[4])
    BF_orb_det(img_ubc[0], img_ubc[5])

    # SIFT BF
    BF_sift_det(img_ubc[0], img_ubc[1])
    BF_sift_det(img_ubc[0], img_ubc[2])
    BF_sift_det(img_ubc[0], img_ubc[3])
    BF_sift_det(img_ubc[0], img_ubc[4])
    BF_sift_det(img_ubc[0], img_ubc[5])

    # AKAZE BF
    AKAZE_BF(img_ubc[0], img_ubc[1])
    AKAZE_BF(img_ubc[0], img_ubc[2])
    AKAZE_BF(img_ubc[0], img_ubc[3])
    AKAZE_BF(img_ubc[0], img_ubc[4])
    AKAZE_BF(img_ubc[0], img_ubc[5])

    # SIFT FLANN
    feat_match_FLANN_sift_visualize(img_ubc[0], img_ubc[1])
    feat_match_FLANN_sift_visualize(img_ubc[0], img_ubc[2])
    feat_match_FLANN_sift_visualize(img_ubc[0], img_ubc[3])
    feat_match_FLANN_sift_visualize(img_ubc[0], img_ubc[4])
    feat_match_FLANN_sift_visualize(img_ubc[0], img_ubc[5])

    # SURF FLANN
    feat_match_FLANN_surf_visualize(img_ubc[0], img_ubc[1])
    feat_match_FLANN_surf_visualize(img_ubc[0], img_ubc[2])
    feat_match_FLANN_surf_visualize(img_ubc[0], img_ubc[3])
    feat_match_FLANN_surf_visualize(img_ubc[0], img_ubc[4])
    feat_match_FLANN_surf_visualize(img_ubc[0], img_ubc[5])


if __name__ == '__main__':
    run()
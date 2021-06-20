# feature matching
import cv2
import numpy as np
import glob
from matplotlib import image
import time


# # BF matcher
# cách này đôi khi có sự nhầm lẫn khi có một số hình ảnh giống nhau
def BF_ORB_det(img1, img2):
    """Calculate evaluation of BF matcher in ORB detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    start_time_ORB_BF = time.time()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    # create BFMatcher object
    BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.95:
            good_matches.append([m1])

    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    correct_matches = len(good_matches)
    correspondences = len(matches)

    print(f"All matches of ORB BF: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall ORB BF: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision ORB BF: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches ORB BF: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision ORB BF: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches ORB BF: {false_matches}")

    end_time_ORB_BF = time.time()
    over_time_ORB_BF = end_time_ORB_BF - start_time_ORB_BF - over_time_append_img
    print("Thời gian thực hiện thuật toán là {:.2f}".format(over_time_ORB_BF))

    print()
    return


# # cách này cho kết quả tốt hơn, ổn định hơn
def BF_SIFT_det(img1, img2):
    """Calculate evaluation of BF matcher in SIFT detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # create our SIFT detector and detect key points and descriptors
    start_time_SIFT_BF = time.time()
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match with BF
    bf = cv2.BFMatcher()

    # knnMatch - visualize descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # apply David Lowe's ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.95 * m2.distance:
            good_matches.append([m1])

    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of BF SIFT: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall BF SIFT: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision ORB BF: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF SIFT: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision BF SIFT: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches BF SIFT: {false_matches}")

    end_time_SIFT_BF = time.time()
    over_time_sift_bf = end_time_SIFT_BF - start_time_SIFT_BF - over_time_append_img
    print("Thời gian thực hiện thuật toán là {:.2f}".format(over_time_sift_bf))

    print()
    return


def AKAZE_BF(img1, img2):
    """Calculate evaluation of BF matcher in AKAZE BF: Recall, 1-precision, precision, correct matches,
    false matches"""
    # load the image and convert it to grayscale
    start_time_AKAZE_BF = time.time()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kp1, des1) = detector.detectAndCompute(gray1, None)
    (kp2, des2) = detector.detectAndCompute(gray2, None)
    #
    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of BF AKAZE: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall BF AKAZE: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision BF AKAZE: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF AKAZE: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision BF AKAZE: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches BF AKAZE: {false_matches}")

    end_time_AKAZE_BF = time.time()
    over_time_AKAZE_BF = end_time_AKAZE_BF - start_time_AKAZE_BF - - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_AKAZE_BF))

    print()
    return


def KAZE_BF(img1, img2):
    """Calculate evaluation of BF matcher in AKAZE BF: Recall, 1-precision, precision, correct matches,
    false matches"""
    # load the image and convert it to grayscale
    start_time_AKAZE_BF = time.time()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # initialize the KAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.KAZE_create()
    (kp1, des1) = detector.detectAndCompute(gray1, None)
    (kp2, des2) = detector.detectAndCompute(gray2, None)
    #
    print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))

    # Match the features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of BF KAZE: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall BF KAZE: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision BF KAZE: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF KAZE: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision BF KAZE: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches BF KAZE: {false_matches}")

    end_time_AKAZE_BF = time.time()
    over_time_AKAZE_BF = end_time_AKAZE_BF - start_time_AKAZE_BF - - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_AKAZE_BF))

    print()
    return


def BRISK_BF(img1, img2):
    """Calculate evaluation of BF matcher in BRISK detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # load the image and convert it to grayscale
    start_time_BRISK_BF = time.time()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate BRISK descriptor
    BRISK = cv2.BRISK_create()

    # Find the keypoints and compute the descriptors for input and training-set image
    kp1, des1 = BRISK.detectAndCompute(gray1, None)
    kp2, des2 = BRISK.detectAndCompute(gray2, None)

    # create BFMatcher object
    BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.95 * m2.distance:
            good_matches.append([m1])

    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of BF BRISK: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall BF BRISK: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision BF BRISK: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF BRISK: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision BF BRISK: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches BF BRISK: {false_matches}")

    end_time_BRISK_BF = time.time()
    over_time_BRISK_BF = end_time_BRISK_BF - start_time_BRISK_BF - - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_BRISK_BF))

    print()
    return


####################################################################


# # visualize FLANN
def feat_match_FLANN_SIFT_visualize(img1, img2):
    """Calculate evaluation of FLANN matcher in SIFT detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # Initiate SIFT detector
    start_time_SIFT_FLANN = time.time()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # create new object
    matchesMask = [[0, 0] for i in range(len(matches))]

    # check if it is a good match
    good_match = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.95 * m2.distance:
            matchesMask[i] = [1, 0]
            good_match.append(matchesMask[i])

    correct_matches = len(good_match)
    correspondences = len(matches)
    print(f"All matches of FLANN SIFT: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall FLANN SIFT: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision FLANN SIFT: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches FLANN SIFT: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision FLANN SIFT: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches FLANN SIFT: {false_matches}")

    end_time_SIFT_FLANN = time.time()
    over_time_SIFT_FLANN = end_time_SIFT_FLANN - start_time_SIFT_FLANN - - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_SIFT_FLANN))
    print()
    return


def feat_match_FLANN_SURF_visualize(img1, img2):
    """Calculate evaluation of FLANN matcher in SURF detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # Initiate SIFT detector
    start_time_SURF_FLANN = time.time()
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    # -- Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    correct_matches = len(good_matches)
    correspondences = len(knn_matches)
    print(f"All matches of SURF FLANN: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall SURF FLANN: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision SURF FLANN: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches SURF FLANN: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision SURF FLANN: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches SURF FLANN: {false_matches}")

    end_time_SURF_FLANN = time.time()
    over_time_SURF_FLANN = end_time_SURF_FLANN - start_time_SURF_FLANN - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_SURF_FLANN))

    print()
    return


def feat_match_BF_SURF_visualize(img1, img2):
    """Calculate evaluation of BF matcher in SURF detector: Recall, 1-precision, precision, correct matches,
    false matches"""
    # Initiate SIFT detector
    start_time_SURF_BF = time.time()
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=500)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    BFMatcher = cv2.BFMatcher()
    matches = BFMatcher.knnMatch(des1, des2, k=2)

    # -- Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of SURF BF: {correspondences}")

    # Good ratio
    recall = float(correct_matches / correspondences)
    print("Recall SURF BF: {:.2f}".format(recall))

    # Bad ratio
    precision_1 = float((correspondences - correct_matches) / correspondences)
    print("1-precision SURF BF: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches SURF BF: {correct_matches}")

    # Dispersion ratio
    precision = (1 - precision_1)
    print("Precision SURF BF: {:.2f}".format(precision))

    false_matches = round(correspondences * recall * precision_1 / precision)
    print(f"False matches SURF BF: {false_matches}")

    end_time_SURF_BF = time.time()
    over_time_SURF_FLANN = end_time_SURF_BF - start_time_SURF_BF - over_time_append_img
    print("Thời gian thực hiện thuật toán là: {:.2f}".format(over_time_SURF_FLANN))

    print()
    return


# change view points of an image
def change_viewpoints_img(img):
    # Locate points of the documents or object which you want to transform
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    # Apply Perspective Transform Algorithm
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Wrap the transformed image
    dst = cv2.warpPerspective(img, M, (300, 300))
    return dst


# change brightness of an image
def change_brightness_img(img, alpha=1.0, beta=100):
    new_image = np.zeros(img.shape, img.dtype)
    # alpha = 1.0  # Simple contrast control
    # beta = 100  # Simple brightness control
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
    return new_image


# blur image
def blur_img(img, size, sigma):
    return cv2.GaussianBlur(np.float32(img), size, sigma)


# rotate image
def rotate_img(img, size):
    return cv2.rotate(img, size)


# resize image
def resize_img(img, size):
    return cv2.resize(img, size)


# jpeg compression image
def jpeg_compress_img(img, jpeg_quality=95):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, encimg = cv2.imencode('.ppm', img, encode_param)

    # decode from jpeg format
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def append_image_in_folder(path):
    """Return to an image in specified folder(images must be ppm files)"""
    images = []
    for filename in glob.glob(path + '*.ppm'):
        # load image
        img_data = image.imread(filename)
        images.append(img_data)
    return images


start_time_append_img = time.time()
img_bark = append_image_in_folder('img/bark/')
img_bikes = append_image_in_folder('img/bikes/')
img_leuven = append_image_in_folder('img/leuven/')
img_trees = append_image_in_folder('img/trees/')
img_ubc = append_image_in_folder('img/ubc/')
img_wall = append_image_in_folder('img/wall/')
img_graf = append_image_in_folder('img/graf/')
end_time_append_img = time.time()
over_time_append_img = (end_time_append_img - start_time_append_img)
print("Thời gian thực hiện thuật toán thêm hình ảnh vào thư mục có sẵn là {:.2f}s".format(over_time_append_img))


# read image
def run():
    """Run the program"""
    # ORB BF
    BF_ORB_det(img_bark[0], img_bark[1])
    BF_ORB_det(img_bark[0], img_bark[2])
    BF_ORB_det(img_bark[0], img_bark[3])
    BF_ORB_det(img_bark[0], img_bark[4])
    BF_ORB_det(img_bark[0], img_bark[5])

    # SIFT BF
    BF_SIFT_det(img_bark[0], img_bark[1])
    BF_SIFT_det(img_bark[0], img_bark[2])
    BF_SIFT_det(img_bark[0], img_bark[3])
    BF_SIFT_det(img_bark[0], img_bark[4])
    BF_SIFT_det(img_bark[0], img_bark[5])

    # SIFT FLANN
    feat_match_FLANN_SIFT_visualize(img_bark[0], img_bark[1])
    feat_match_FLANN_SIFT_visualize(img_bark[0], img_bark[2])
    feat_match_FLANN_SIFT_visualize(img_bark[0], img_bark[3])
    feat_match_FLANN_SIFT_visualize(img_bark[0], img_bark[4])
    feat_match_FLANN_SIFT_visualize(img_bark[0], img_bark[5])

    # SUFT FLANN
    feat_match_FLANN_SURF_visualize(img_bark[0], img_bark[1])
    feat_match_FLANN_SURF_visualize(img_bark[0], img_bark[2])
    feat_match_FLANN_SURF_visualize(img_bark[0], img_bark[3])
    feat_match_FLANN_SURF_visualize(img_bark[0], img_bark[4])
    feat_match_FLANN_SURF_visualize(img_bark[0], img_bark[5])

    # AKAZE BF
    AKAZE_BF(img_bark[0], img_bark[1])
    AKAZE_BF(img_bark[0], img_bark[2])
    AKAZE_BF(img_bark[0], img_bark[3])
    AKAZE_BF(img_bark[0], img_bark[4])
    AKAZE_BF(img_bark[0], img_bark[5])

    # BRISK BF
    BRISK_BF(img_bark[0], img_bark[1])
    BRISK_BF(img_bark[0], img_bark[2])
    BRISK_BF(img_bark[0], img_bark[3])
    BRISK_BF(img_bark[0], img_bark[4])
    BRISK_BF(img_bark[0], img_bark[5])

    # SURF BF
    feat_match_BF_SURF_visualize(img_bark[0], img_bark[1])
    feat_match_BF_SURF_visualize(img_bark[0], img_bark[2])
    feat_match_BF_SURF_visualize(img_bark[0], img_bark[3])
    feat_match_BF_SURF_visualize(img_bark[0], img_bark[4])
    feat_match_BF_SURF_visualize(img_bark[0], img_bark[5])
    
    # KAZE BF 
    KAZE_BF(img_bark[0], img_bark[1])
    KAZE_BF(img_bark[0], img_bark[2])
    KAZE_BF(img_bark[0], img_bark[3])
    KAZE_BF(img_bark[0], img_bark[4])
    KAZE_BF(img_bark[0], img_bark[5])


if __name__ == '__main__':
    run()

# feature matching
import cv2
import glob


# # BF matcher
# cách này đôi khi có sự nhầm lẫn khi có một số hình ảnh giống nhau
def append_image_in_folder(path):
    """Return to an image in specified folder(images must be ppm files)"""
    images = []
    for filename in glob.glob(path + '*.ppm'):
        # load image
        img_data = cv2.imread(filename)
        images.append(img_data)
    return images


def BF_ORB_keypoint(img1, img2):
    """Return length of keypoint1 and keypoint2 BF ORB"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1 = orb.detect(gray1, None)
    kp2 = orb.detect(gray2, None)
    return [len(kp1), len(kp2)]


def BF_ORB_matches(img1, img2):
    """Return all good matches BF ORB"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create()

    # # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    # create BFMatcher object
    BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.knnMatch(des1, des2, 2)

    return matches


def BF_ORB_good_matches(img1, img2):
    """Return good matches from all of good matches BF ORB"""
    matches = BF_ORB_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_ORB_correspondences(img1, img2):
    """Return all of good matches BF ORB"""
    matches = BF_ORB_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_ORB_recall(img1, img2):
    """Return recall of BF ORB"""
    good_matches = BF_ORB_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_ORB_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_ORB_correct_matches(img1, img2):
    """Return correct matches of BF ORB"""
    correspondences = BF_ORB_correspondences(img1, img2)
    recall = BF_ORB_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_ORB_precision_1(img1, img2):
    """Return 1-precision of BF ORB"""
    correspondences = BF_ORB_correspondences(img1, img2)
    correct_matches = BF_ORB_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_ORB_precision(img1, img2):
    """Return precision of BF ORB"""
    precision_1 = BF_ORB_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_ORB_false_matches(img1, img2):
    """Return false matches of BF ORB"""
    correspondences = BF_ORB_correspondences(img1, img2)
    recall = BF_ORB_recall(img1, img2)
    precision_1 = BF_ORB_precision_1(img1, img2)
    precision = BF_ORB_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches


def BF_AKAZE_keypoint(img1, img2):
    """Return length of keypoint BF AKAZE"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()

    # Find the key points and descriptors with SIFT
    kp1 = akaze.detect(gray1, None)
    kp2 = akaze.detect(gray2, None)

    return [len(kp1), len(kp2)]


def BF_AKAZE_matches(img1, img2):
    """Return all matches(correspondences) of BF AKAZE"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    return matches


def BF_AKAZE_good_matches(img1, img2):
    """Return good matches from all of good matches BF AKAZE"""
    matches = BF_AKAZE_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_AKAZE_correspondences(img1, img2):
    """Return all of good matches BF AKAZE"""
    matches = BF_AKAZE_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_AKAZE_recall(img1, img2):
    """Return recall of BF BF AKAZE"""
    good_matches = BF_AKAZE_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_AKAZE_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_AKAZE_correct_matches(img1, img2):
    """Return correct matches of BF AKAZE"""
    correspondences = BF_AKAZE_correspondences(img1, img2)
    recall = BF_AKAZE_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_AKAZE_precision_1(img1, img2):
    """Return 1-precision of BF AKAZE"""
    correspondences = BF_AKAZE_correspondences(img1, img2)
    correct_matches = BF_AKAZE_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_AKAZE_precision(img1, img2):
    """Return precision of BF AKAZE"""
    precision_1 = BF_AKAZE_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_AKAZE_false_matches(img1, img2):
    """Return false matches of BF AKAZE"""
    correspondences = BF_AKAZE_correspondences(img1, img2)
    recall = BF_AKAZE_recall(img1, img2)
    precision_1 = BF_AKAZE_precision_1(img1, img2)
    precision = BF_AKAZE_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches


def BF_KAZE_keypoint(img1, img2):
    """Return length of keypoint BF KAZE"""
    # create our KAZE detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kaze = cv2.KAZE_create()

    # Find the key points and descriptors with KAZE
    kp1 = kaze.detect(gray1, None)
    kp2 = kaze.detect(gray2, None)

    return [len(kp1), len(kp2)]


def BF_KAZE_matches(img1, img2):
    """Return all matches(correspondences) of BF AKAZE"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kaze = cv2.KAZE_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = kaze.detectAndCompute(gray1, None)
    kp2, des2 = kaze.detectAndCompute(gray2, None)

    # Match the features
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    return matches


def BF_KAZE_good_matches(img1, img2):
    """Return good matches from all of good matches BF AKAZE"""
    matches = BF_KAZE_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_KAZE_correspondences(img1, img2):
    """Return all of good matches BF AKAZE"""
    matches = BF_KAZE_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_KAZE_recall(img1, img2):
    """Return recall of BF BF KAZE"""
    good_matches = BF_KAZE_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_KAZE_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_KAZE_correct_matches(img1, img2):
    """Return correct matches of BF KAZE"""
    correspondences = BF_KAZE_correspondences(img1, img2)
    recall = BF_KAZE_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_KAZE_precision_1(img1, img2):
    """Return 1-precision of BF KAZE"""
    correspondences = BF_KAZE_correspondences(img1, img2)
    correct_matches = BF_KAZE_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_KAZE_precision(img1, img2):
    """Return precision of BF AKAZE"""
    precision_1 = BF_KAZE_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_KAZE_false_matches(img1, img2):
    """Return false matches of BF AKAZE"""
    correspondences = BF_KAZE_correspondences(img1, img2)
    recall = BF_KAZE_recall(img1, img2)
    precision_1 = BF_KAZE_precision_1(img1, img2)
    precision = BF_KAZE_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches


def BF_BRISK_keypoint(img1, img2):
    """Return length of keypoint BF BRISK"""
    # create our KAZE detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    brisk = cv2.BRISK_create()

    # Find the key points and descriptors with KAZE
    kp1 = brisk.detect(gray1, None)
    kp2 = brisk.detect(gray2, None)

    return [len(kp1), len(kp2)]


def BF_BRISK_matches(img1, img2):
    """Return all matches(correspondences) of BF BRISK"""
    # create our SIFT detector and detect key points and descriptors
    brisk = cv2.BRISK_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)  # typo fixed

    return matches


def BF_BRISK_good_matches(img1, img2):
    """Return good matches from all of good matches BF BRISK"""
    matches = BF_BRISK_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_BRISK_correspondences(img1, img2):
    """Return all of good matches BF BRISK"""
    matches = BF_BRISK_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_BRISK_recall(img1, img2):
    """Return recall of BF BF BRISK"""
    good_matches = BF_BRISK_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_BRISK_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_BRISK_correct_matches(img1, img2):
    """Return correct matches of BF BRISK"""
    correspondences = BF_BRISK_correspondences(img1, img2)
    recall = BF_BRISK_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_BRISK_precision_1(img1, img2):
    """Return 1-precision of BF BRISK"""
    correspondences = BF_BRISK_correspondences(img1, img2)
    correct_matches = BF_BRISK_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_BRISK_precision(img1, img2):
    """Return precision of BF BRISK"""
    precision_1 = BF_BRISK_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_BRISK_false_matches(img1, img2):
    """Return false matches of BF BRISK"""
    correspondences = BF_BRISK_correspondences(img1, img2)
    recall = BF_BRISK_recall(img1, img2)
    precision_1 = BF_BRISK_precision_1(img1, img2)
    precision = BF_BRISK_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches


# cách này cho kết quả tốt hơn, ổn định hơn
def BF_SIFT_keypoint(img1, img2):
    """Return length of keypoint BF SIFT"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)

    return [len(kp1), len(kp2)]


def BF_SIFT_matches(img1, img2):
    """Return all matches(correspondences) of BF SIFT"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match with BF
    bf = cv2.BFMatcher()

    # knnMatch - visualize descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    return matches


def BF_SIFT_good_matches(img1, img2):
    """Return good matches from all of good matches BF SIFT"""
    matches = BF_SIFT_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_SIFT_correspondences(img1, img2):
    """Return all of good matches BF SIFT"""
    matches = BF_SIFT_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_SIFT_recall(img1, img2):
    """Return recall of BF BF SIFT"""
    good_matches = BF_SIFT_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_SIFT_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_SIFT_correct_matches(img1, img2):
    """Return correct matches of BF SIFT"""
    correspondences = BF_SIFT_correspondences(img1, img2)
    recall = BF_SIFT_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_SIFT_precision_1(img1, img2):
    """Return 1-precision of BF SIFT"""
    correspondences = BF_SIFT_correspondences(img1, img2)
    correct_matches = BF_SIFT_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_SIFT_precision(img1, img2):
    """Return precision of BF SIFT"""
    precision_1 = BF_SIFT_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_SIFT_false_matches(img1, img2):
    """Return false matches of BF SIFT"""
    correspondences = BF_SIFT_correspondences(img1, img2)
    recall = BF_SIFT_recall(img1, img2)
    precision_1 = BF_SIFT_precision_1(img1, img2)
    precision = BF_SIFT_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches


def BF_SURF_keypoint(img1, img2):
    """Return length of keypoint BF SURF"""
    # create our SIFT detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # Find the key points and descriptors with SIFT
    kp1 = surf.detect(gray1, None)
    kp2 = surf.detect(gray2, None)

    return [len(kp1), len(kp2)]


def BF_SURF_matches(img1, img2):
    """Return all matches(correspondences) of BF SURF"""
    # create our SURF detector and detect key points and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # Find the key points and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)

    # Match with BF
    bf = cv2.BFMatcher()

    # knnMatch - visualize descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    return matches


def BF_SURF_good_matches(img1, img2):
    """Return good matches from all of good matches BF SURF"""
    matches = BF_SURF_matches(img1, img2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * 0.9:
            good_matches.append([m1])
    return good_matches


def BF_SURF_correspondences(img1, img2):
    """Return all of good matches BF SURF"""
    matches = BF_SURF_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def BF_SURF_recall(img1, img2):
    """Return recall of BF BF SURF"""
    good_matches = BF_SURF_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = BF_SURF_correspondences(img1, img2)
    # Good ratio
    recall = float(correct_matches / correspondences)
    return recall


def BF_SURF_correct_matches(img1, img2):
    """Return correct matches of BF SURF"""
    correspondences = BF_SURF_correspondences(img1, img2)
    recall = BF_SURF_recall(img1, img2)
    correct_matches = round(correspondences * recall)
    return correct_matches


def BF_SURF_precision_1(img1, img2):
    """Return 1-precision of BF SURF"""
    correspondences = BF_SURF_correspondences(img1, img2)
    correct_matches = BF_SURF_correct_matches(img1, img2)
    precision_1 = float((correspondences - correct_matches) / correspondences)
    return precision_1


def BF_SURF_precision(img1, img2):
    """Return precision of BF SURF"""
    precision_1 = BF_SURF_precision_1(img1, img2)
    precision = 1 - precision_1
    return precision


def BF_SURF_false_matches(img1, img2):
    """Return false matches of BF SURF"""
    correspondences = BF_SURF_correspondences(img1, img2)
    recall = BF_SURF_recall(img1, img2)
    precision_1 = BF_SURF_precision_1(img1, img2)
    precision = BF_SURF_precision(img1, img2)
    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)
    return false_matches
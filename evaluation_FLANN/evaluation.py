import cv2


def FLANN_SIFT_keypoint(img1, img2):
    """Return length of keypoint1 and keypoint2 FLANN SIFT"""
    # convert image to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)

    return [len(kp1), len(kp2)]


def FLANN_SIFT_matches(img1, img2):
    """Return all good matches FLANN SIFT"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    return matches


def FLANN_SIFT_good_matches(img1, img2):
    """Return good matches from all of good matches FLANN SIFT"""
    matches = FLANN_SIFT_matches(img1, img2)

    # create new object
    matchesMask = [[0, 0] for i in range(len(matches))]

    # check if it is a good match
    good_match = []

    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.9 * m2.distance:
            matchesMask[i] = [1, 0]
            good_match.append(matchesMask[i])

    return good_match


def FLANN_SIFT_correspondences(img1, img2):
    """Return all of good matches FLANN SIFT"""
    matches = FLANN_SIFT_matches(img1, img2)
    correspondences = len(matches)
    return correspondences


def FLANN_SIFT_recall(img1, img2):
    """Return recall of FLANN SIFT"""
    good_matches = FLANN_SIFT_good_matches(img1, img2)
    correct_matches = len(good_matches)

    correspondences = FLANN_SIFT_correspondences(img1, img2)

    # Good ratio
    recall = float(correct_matches / correspondences)

    return recall


def FLANN_SIFT_correct_matches(img1, img2):
    """Return correct matches of FLANN SIFT"""
    correspondences = FLANN_SIFT_correspondences(img1, img2)
    recall = FLANN_SIFT_recall(img1, img2)

    correct_matches = round(correspondences * recall)

    return correct_matches


def FLANN_SIFT_precision_1(img1, img2):
    """Return 1-precision of FLANN SIFT"""
    correspondences = FLANN_SIFT_correspondences(img1, img2)
    correct_matches = FLANN_SIFT_correct_matches(img1, img2)

    precision_1 = float((correspondences - correct_matches) / correspondences)

    return precision_1


def FLANN_SIFT_precision(img1, img2):
    """Return precision of FLANN SIFT"""
    precision_1 = FLANN_SIFT_precision_1(img1, img2)

    precision = 1 - precision_1

    return precision


def FLANN_SIFT_false_matches(img1, img2):
    """Return false matches of FLANN SIFT"""
    correspondences = FLANN_SIFT_correspondences(img1, img2)
    recall = FLANN_SIFT_recall(img1, img2)
    precision_1 = FLANN_SIFT_precision_1(img1, img2)
    precision = FLANN_SIFT_precision(img1, img2)

    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)

    return false_matches


def FLANN_SURF_keypoint(img1, img2):
    """Return length of keypoint1 and keypoint2 FLANN SURF"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # find the keypoints and descriptors with SIFT
    kp1 = surf.detect(gray1, None)
    kp2 = surf.detect(gray2, None)

    return [len(kp1), len(kp2)]


def FLANN_SURF_matches(img1, img2):
    """Return all good matches FLANN SURF"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    matches = matcher.knnMatch(des1, des2, 2)

    return matches


def FLANN_SURF_good_matches(img1, img2):
    """Return good matches from all of good matches FLANN SURF"""
    matches = FLANN_SURF_matches(img1, img2)

    # check if it is a good match
    good_match = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_match.append(m)

    return good_match


def FLANN_SURF_correspondences(img1, img2):
    """Return all of good matches FLANN SURF"""
    matches = FLANN_SURF_good_matches(img1, img2)

    correspondences = len(matches)

    return correspondences


def FLANN_SURF_recall(img1, img2):
    """Return recall of FLANN SIFT"""
    good_matches = FLANN_SURF_good_matches(img1, img2)
    correct_matches = len(good_matches)
    correspondences = FLANN_SURF_correspondences(img1, img2)

    # Good ratio
    recall = float(correct_matches / correspondences)

    return recall


def FLANN_SURF_correct_matches(img1, img2):
    """Return correct matches of FLANN SIFT"""
    correspondences = FLANN_SURF_correspondences(img1, img2)
    recall = FLANN_SURF_recall(img1, img2)

    correct_matches = round(correspondences * recall)

    return correct_matches


def FLANN_SURF_precision_1(img1, img2):
    """Return 1-precision of FLANN SIFT"""
    correspondences = FLANN_SURF_correspondences(img1, img2)
    correct_matches = FLANN_SURF_correct_matches(img1, img2)

    precision_1 = float((correspondences - correct_matches) / correspondences)

    return precision_1


def FLANN_SURF_precision(img1, img2):
    """Return precision of FLANN SIFT"""
    precision_1 = FLANN_SURF_precision_1(img1, img2)

    precision = 1 - precision_1

    return precision


def FLANN_SURF_false_matches(img1, img2):
    """Return false matches of FLANN SIFT"""
    correspondences = FLANN_SURF_correspondences(img1, img2)
    recall = FLANN_SURF_recall(img1, img2)
    precision_1 = FLANN_SURF_precision_1(img1, img2)
    precision = FLANN_SURF_precision(img1, img2)

    if precision == 0:
        false_matches = correspondences
    else:
        false_matches = round(correspondences * recall * precision_1 / precision)

    return false_matches

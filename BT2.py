# feature matching
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import image


# # BF matcher
# cách này đôi khi có sự nhầm lẫn khi có một số hình ảnh giống nhau
def BF_orb_det(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    # print(f"Des1: {len(descriptors1)}")
    # print(f"Des2: {len(descriptors2)}")
    # print(f"Kp1: {len(keypoints1)}")
    # print(f"Kp2: {len(keypoints2)}")

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # best matches in 2 images
    matches = bf.match(descriptors1, descriptors2)
    # print(f"matches distance: {len(matches)}")

    # sort to get the lowest distance
    # matches = sorted(matches, key=lambda x: x.distance)

    # draw 10 matches
    # ORB_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None,
    #                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    correct_matches = len(matches[:10])
    correspondences = len(matches)

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
    # create our SIFT detector and detect key points and descriptors
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match with BF
    bf = cv2.BFMatcher()

    # print descriptors -
    # print(des1)
    # print(des2)

    # knnMatch - visualize descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # extract one first and one second-best match and compare there distance measurements
    # AA1 = matches[131][0]
    # print(AA1.distance)
    # AA2 = matches[131][1]
    # print(AA2.distance)

    # if the first best match is pretty close to the second match, then this point probably not distinct enough
    # BB1 = matches[1][0]
    # print(BB1.distance)
    # BB2 = matches[1][1]
    # print(BB2.distance)

    # apply David Lowe's ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append([m1])

    # print(good_matches)

    # print length of matches and good matches
    # print(f"Matches: {len(matches)}")
    # print(f"Good match: {len(good_matches)}")
    correct_matches = len(good_matches)
    correspondences = len(matches)
    print(f"All matches of BF SIFT: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall BF SIFT: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision ORB BF: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches BF SIFT: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision BF SIFT: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches BF SIFT: {false_matches}")
    print()
    # draw these matches and see how they performed
    SIFT_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

    return


####################################################################
# # # FLANN matcher basic
# def feat_match_FLANN_sift_basic(img1, img2):
#     # detect feature with SIFT
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     # find key points and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)
#
#     # use k-dimensional tree for organizing points
#     FLAN_INDEX_TREE = 0
#     index_params = dict(algorithm=FLAN_INDEX_TREE, trees=5)
#     search_params = dict(checks=100)
#
#     # create FLANN object and search params
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#     # calculate matches
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     # apply David Lowe's ratio test
#     good_matches = []
#
#     for m1, m2 in matches:
#         if m1.distance < 0.5 * m2.distance:
#             good_matches.append([m1])
#
#     correct_matches = len(good_matches)
#     correspondences = len(matches)
#     print(f"All matches of ORB BF: {correspondences}")
#
#     recall = float(correct_matches / correspondences)
#     print("Recall ORB BF: {:.2f}".format(recall))
#
#     precision_1 = float(correspondences / (correct_matches + correspondences))
#     print("1-precision ORB BF: {:.2f}".format(precision_1))
#
#     correct_matches = round(correspondences * recall)
#     print(f"Correct matches ORB BF: {correct_matches}")
#
#     # false_matches = correspondences - correct_matches
#     precision = (1 - precision_1)
#     print("Precision ORB BF: {:.2f}".format(precision))
#
#     false_matches = round(correspondences - correct_matches)
#     print(f"False matches ORB BF: {false_matches}")
#     print()
#     # draw only lines between matching points
#     # flann_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
#
#     # all points will be show
#     flann_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=0)
#     return


# # visualize FLANN
def feat_match_FLANN_sift_visualize(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # create new object
    matchesMask = [[0, 0] for i in range(len(matches))]

    # check if it is a good match
    good_match = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.7 * m2.distance:
            matchesMask[i] = [1, 0]
            good_match.append(matchesMask[i])

    # create a drawing params dictionary
    draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(0, 255, 0), matchesMask=matchesMask, flags=0)

    # draw matches base on color
    flann_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    correct_matches = len(good_match)
    correspondences = len(matches)
    print(f"All matches of FLANN SIFT: {correspondences}")

    recall = float(correct_matches / correspondences)
    print("Recall FLANN SIFT: {:.2f}".format(recall))

    precision_1 = float(correspondences / (correct_matches + correspondences))
    print("1-precision FLANN SIFT: {:.2f}".format(precision_1))

    correct_matches = round(correspondences * recall)
    print(f"Correct matches FLANN SIFT: {correct_matches}")

    # false_matches = correspondences - correct_matches
    precision = (1 - precision_1)
    print("Precision ORB BF: {:.2f}".format(precision))

    false_matches = round(correspondences - correct_matches)
    print(f"False matches ORB BF: {false_matches}")
    print()
    return


def feat_match_FLANN_surf_visualize(img1, img2):
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    correct_matches = len(good_matches)
    correspondences = len(knn_matches)
    print(f"All matches of ORB BF: {correspondences}")

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
    images = []
    for filename in glob.glob(path + '*.ppm'):
        # load image
        img_data = image.imread(filename)
        images.append(img_data)
    return images


# read image
img_bark = append_image_in_folder('img/bark/')
# blur = blur_img(img, (5, 5), 7)
# rotate = rotate_img(img, cv2.ROTATE_90_CLOCKWISE)
# resize = resize_img(img, (700, 700))
# jpeg = jpeg_compress_img(img)
# bright = change_brightness_img(img)
# viewpoints = change_viewpoints_img(img)
# print(img_bark[0].shape)
# matching bf
BF_orb_det(img_bark[0], img_bark[1])
BF_orb_det(img_bark[0], img_bark[2])
BF_orb_det(img_bark[0], img_bark[3])
BF_orb_det(img_bark[0], img_bark[4])
BF_orb_det(img_bark[0], img_bark[5])

# bf_orb_rotate = BF_orb_det(img, rotate)
# bf_orb_resize = BF_orb_det(img, resize)
# bf_orb_jpeg = BF_orb_det(img, jpeg)
# bf_orb_bright = BF_orb_det(img, bright)
# bf_orb_viewpoints = BF_orb_det(img, viewpoints)
# cv2.imshow("orb_1", bf_orb_1)
# cv2.imshow("orb_2", bf_orb_2)
# cv2.imshow("orb_3", bf_orb_3)
# cv2.imshow("orb_4", bf_orb_4)
# cv2.imshow("orb_5", bf_orb_5)


# matching bf
# SIFT
BF_sift_det(img_bark[0], img_bark[1])
BF_sift_det(img_bark[0], img_bark[2])
BF_sift_det(img_bark[0], img_bark[3])
BF_sift_det(img_bark[0], img_bark[4])
BF_sift_det(img_bark[0], img_bark[5])


# matching FLANN
# # SIFT Basic
# flann_sift_basic_1 = feat_match_FLANN_sift_basic(img_bark[0], img_bark[1])
# flann_sift_basic_2 = feat_match_FLANN_sift_basic(img_bark[0], img_bark[2])
# flann_sift_basic_3 = feat_match_FLANN_sift_basic(img_bark[0], img_bark[3])
# flann_sift_basic_4 = feat_match_FLANN_sift_basic(img_bark[0], img_bark[4])
# flann_sift_basic_5 = feat_match_FLANN_sift_basic(img_bark[0], img_bark[5])
#
# print(flann_sift_basic_1)
# print(flann_sift_basic_2)
# print(flann_sift_basic_3)
# print(flann_sift_basic_4)
# print(flann_sift_basic_5)

# # SIFT Visual
feat_match_FLANN_sift_visualize(img_bark[0], img_bark[1])
feat_match_FLANN_sift_visualize(img_bark[0], img_bark[2])
feat_match_FLANN_sift_visualize(img_bark[0], img_bark[3])
feat_match_FLANN_sift_visualize(img_bark[0], img_bark[4])
feat_match_FLANN_sift_visualize(img_bark[0], img_bark[5])

feat_match_FLANN_surf_visualize(img_bark[0], img_bark[1])
feat_match_FLANN_surf_visualize(img_bark[0], img_bark[2])
feat_match_FLANN_surf_visualize(img_bark[0], img_bark[3])
feat_match_FLANN_surf_visualize(img_bark[0], img_bark[4])
feat_match_FLANN_surf_visualize(img_bark[0], img_bark[5])

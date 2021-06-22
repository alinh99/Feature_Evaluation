from evaluation import BF_ORB_precision_1, BF_ORB_precision, BF_ORB_recall, BF_AKAZE_precision_1, BF_KAZE_recall, \
    BF_AKAZE_precision, BF_AKAZE_recall, BF_KAZE_precision_1, BF_SURF_precision_1, BF_SURF_precision, BF_SURF_recall, \
    BF_SIFT_recall, BF_SIFT_precision, BF_SIFT_precision_1, BF_BRISK_recall, BF_BRISK_precision, BF_KAZE_precision, \
    BF_BRISK_precision_1, append_image_in_folder
import matplotlib.pyplot as plt
path = "E:\\VNUKUniversity\\ThirdYear\\Internship\\FeatureExercises\\img\\bikes\\"
img = append_image_in_folder(path)


def BF_SURF_average_precision_1():
    """Return average of 1 - precision BF SURF"""
    precision_1_1 = BF_SURF_precision_1(img[0], img[1])
    precision_1_2 = BF_SURF_precision_1(img[0], img[2])
    precision_1_3 = BF_SURF_precision_1(img[0], img[3])
    precision_1_4 = BF_SURF_precision_1(img[0], img[4])
    precision_1_5 = BF_SURF_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def BF_SURF_average_precision():
    """Return average recall of BF SURF"""
    precision1 = BF_SURF_precision(img[0], img[1])
    precision2 = BF_SURF_precision(img[0], img[2])
    precision3 = BF_SURF_precision(img[0], img[3])
    precision4 = BF_SURF_precision(img[0], img[4])
    precision5 = BF_SURF_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_SURF_average_recall():
    """Return average recall of BF SURF"""
    recall1 = BF_SURF_recall(img[0], img[1])
    recall2 = BF_SURF_recall(img[0], img[2])
    recall3 = BF_SURF_recall(img[0], img[3])
    recall4 = BF_SURF_recall(img[0], img[4])
    recall5 = BF_SURF_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)

    return average_recall


def BF_SIFT_average_recall():
    """Return average recall of BF SIFT"""
    recall1 = BF_SIFT_recall(img[0], img[1])
    recall2 = BF_SIFT_recall(img[0], img[2])
    recall3 = BF_SIFT_recall(img[0], img[3])
    recall4 = BF_SIFT_recall(img[0], img[4])
    recall5 = BF_SIFT_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)

    return average_recall


def BF_SIFT_average_precision():
    """Return average recall of BF SIFT"""
    precision1 = BF_SIFT_precision(img[0], img[1])
    precision2 = BF_SIFT_precision(img[0], img[2])
    precision3 = BF_SIFT_precision(img[0], img[3])
    precision4 = BF_SIFT_precision(img[0], img[4])
    precision5 = BF_SIFT_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_SIFT_average_precision_1():
    """Return average of 1 - precision BF SIFT"""
    precision_1_1 = BF_SIFT_precision_1(img[0], img[1])
    precision_1_2 = BF_SIFT_precision_1(img[0], img[2])
    precision_1_3 = BF_SIFT_precision_1(img[0], img[3])
    precision_1_4 = BF_SIFT_precision_1(img[0], img[4])
    precision_1_5 = BF_SIFT_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def BF_BRISK_average_recall():
    """Return average recall of BF BRISK"""
    recall1 = BF_BRISK_recall(img[0], img[1])
    recall2 = BF_BRISK_recall(img[0], img[2])
    recall3 = BF_BRISK_recall(img[0], img[3])
    recall4 = BF_BRISK_recall(img[0], img[4])
    recall5 = BF_BRISK_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)
    return average_recall


def BF_BRISK_average_precision():
    """Return average recall of BF BRISK"""
    precision1 = BF_BRISK_precision(img[0], img[1])
    precision2 = BF_BRISK_precision(img[0], img[2])
    precision3 = BF_BRISK_precision(img[0], img[3])
    precision4 = BF_BRISK_precision(img[0], img[4])
    precision5 = BF_BRISK_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_BRISK_average_precision_1():
    """Return average of 1 - precision BF BRISK"""
    precision_1_1 = BF_BRISK_precision_1(img[0], img[1])
    precision_1_2 = BF_BRISK_precision_1(img[0], img[2])
    precision_1_3 = BF_BRISK_precision_1(img[0], img[3])
    precision_1_4 = BF_BRISK_precision_1(img[0], img[4])
    precision_1_5 = BF_BRISK_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def BF_KAZE_average_recall():
    """Return average recall of BF KAZE"""
    recall1 = BF_KAZE_recall(img[0], img[1])
    recall2 = BF_KAZE_recall(img[0], img[2])
    recall3 = BF_KAZE_recall(img[0], img[3])
    recall4 = BF_KAZE_recall(img[0], img[4])
    recall5 = BF_KAZE_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)

    return average_recall


def BF_KAZE_average_precision():
    """Return average recall of BF AKAZE"""
    precision1 = BF_KAZE_precision(img[0], img[1])
    precision2 = BF_KAZE_precision(img[0], img[2])
    precision3 = BF_KAZE_precision(img[0], img[3])
    precision4 = BF_KAZE_precision(img[0], img[4])
    precision5 = BF_KAZE_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_KAZE_average_precision_1():
    """Return average of 1 - precision BF KAZE"""
    precision_1_1 = BF_KAZE_precision_1(img[0], img[1])
    precision_1_2 = BF_KAZE_precision_1(img[0], img[2])
    precision_1_3 = BF_KAZE_precision_1(img[0], img[3])
    precision_1_4 = BF_KAZE_precision_1(img[0], img[4])
    precision_1_5 = BF_KAZE_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def BF_AKAZE_average_recall():
    """Return average recall of BF AKAZE"""
    recall1 = BF_AKAZE_recall(img[0], img[1])
    recall2 = BF_AKAZE_recall(img[0], img[2])
    recall3 = BF_AKAZE_recall(img[0], img[3])
    recall4 = BF_AKAZE_recall(img[0], img[4])
    recall5 = BF_AKAZE_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)
    return average_recall


def BF_AKAZE_average_precision():
    """Return average recall of BF AKAZE"""
    precision1 = BF_AKAZE_precision(img[0], img[1])
    precision2 = BF_AKAZE_precision(img[0], img[2])
    precision3 = BF_AKAZE_precision(img[0], img[3])
    precision4 = BF_AKAZE_precision(img[0], img[4])
    precision5 = BF_AKAZE_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_AKAZE_average_precision_1():
    """Return average of 1 - precision BF AKAZE"""
    precision_1_1 = BF_AKAZE_precision_1(img[0], img[1])
    precision_1_2 = BF_AKAZE_precision_1(img[0], img[2])
    precision_1_3 = BF_AKAZE_precision_1(img[0], img[3])
    precision_1_4 = BF_AKAZE_precision_1(img[0], img[4])
    precision_1_5 = BF_AKAZE_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def BF_ORB_average_recall():
    """Return average recall of BF ORB"""
    recall1 = BF_ORB_recall(img[0], img[1])
    recall2 = BF_ORB_recall(img[0], img[2])
    recall3 = BF_ORB_recall(img[0], img[3])
    recall4 = BF_ORB_recall(img[0], img[4])
    recall5 = BF_ORB_recall(img[0], img[5])

    lst_recall = [recall1, recall2, recall3, recall4, recall5]

    average_recall = sum(lst_recall) / len(lst_recall)
    return average_recall


def BF_ORB_average_precision():
    """Return average recall of BF ORB"""
    precision1 = BF_ORB_precision(img[0], img[1])
    precision2 = BF_ORB_precision(img[0], img[2])
    precision3 = BF_ORB_precision(img[0], img[3])
    precision4 = BF_ORB_precision(img[0], img[4])
    precision5 = BF_ORB_precision(img[0], img[5])

    lst_precision = [precision1, precision2, precision3, precision4, precision5]

    average_precision = sum(lst_precision) / len(lst_precision)

    return average_precision


def BF_ORB_average_precision_1():
    """Return average of 1 - precision BF ORB"""
    precision_1_1 = BF_ORB_precision_1(img[0], img[1])
    precision_1_2 = BF_ORB_precision_1(img[0], img[2])
    precision_1_3 = BF_ORB_precision_1(img[0], img[3])
    precision_1_4 = BF_ORB_precision_1(img[0], img[4])
    precision_1_5 = BF_ORB_precision_1(img[0], img[5])

    lst_precision_1 = [precision_1_1, precision_1_2, precision_1_3, precision_1_4, precision_1_5]

    average_precision_1 = sum(lst_precision_1) / len(lst_precision_1)

    return average_precision_1


def plot_recall():
    """Plot graphs of recall"""
    recall_ORB = BF_ORB_average_recall()
    recall_SIFT = BF_SIFT_average_recall()
    recall_SURF = BF_SURF_average_recall()
    recall_KAZE = BF_KAZE_average_recall()
    recall_AKAZE = BF_AKAZE_average_recall()
    recall_BRISK = BF_BRISK_average_recall()
    recall = {'BF_ORB': recall_ORB, 'BF_SIFT': recall_SIFT, 'BF_SURF': recall_SURF,
              'BF_AKAZE': recall_AKAZE, 'BF_BRISK': recall_BRISK, 'BF_KAZE': recall_KAZE}

    names = list(recall.keys())
    values = list(recall.values())

    plt.figure(figsize=(20, 20))
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)

    plt.suptitle('Recall Plotting in Blur images')

    return plt.show()


def plot_precision():
    """Plot graphs of Precision"""
    precision_ORB = BF_ORB_average_precision()
    precision_SIFT = BF_SIFT_average_precision()
    precision_SURF = BF_SURF_average_precision()
    precision_KAZE = BF_KAZE_average_precision()
    precision_AKAZE = BF_AKAZE_average_precision()
    precision_BRISK = BF_BRISK_average_precision()
    recall = {'BF_ORB': precision_ORB, 'BF_SIFT': precision_SIFT, 'BF_SURF': precision_SURF,
              'BF_AKAZE': precision_AKAZE, 'BF_BRISK': precision_BRISK, 'BF_KAZE': precision_KAZE}

    names = list(recall.keys())
    values = list(recall.values())

    plt.figure(figsize=(20, 20))
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)

    plt.suptitle('Precision Plotting in Blur images')

    return plt.show()


def plot_precision_1():
    """Plot graphs of 1 - precision"""
    precision_1_ORB = BF_ORB_average_precision_1()
    precision_1_SIFT = BF_SIFT_average_precision_1()
    precision_1_SURF = BF_SURF_average_precision_1()
    precision_1_KAZE = BF_KAZE_average_precision_1()
    precision_1_AKAZE = BF_AKAZE_average_precision_1()
    precision_1_BRISK = BF_BRISK_average_precision_1()

    recall = {'BF_ORB': precision_1_ORB, 'BF_SIFT': precision_1_SIFT, 'BF_SURF': precision_1_SURF,
              'BF_AKAZE': precision_1_AKAZE, 'BF_BRISK': precision_1_BRISK, 'BF_KAZE': precision_1_KAZE}

    names = list(recall.keys())
    values = list(recall.values())

    plt.figure(figsize=(20, 20))
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('1 - Precision Plotting in Blur images')

    return plt.show()


plot_recall()
plot_precision()
plot_precision_1()
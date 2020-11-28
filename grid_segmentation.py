import math
import os
import time
from collections import deque, Counter
from statistics import stdev

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from skimage.exposure import histogram
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from threshold_config import ThresholdConfig
from utils.mibi_reader import get_all_point_data
import config.config_settings as config
from utils.utils_functions import mkdir_p


def _split_image_into_quadrants(d, x_0, y_0):
    quadrants = []

    for r in range(0, d.shape[0], int(d.shape[0] / 2)):
        for c in range(0, d.shape[1], int(d.shape[1] / 2)):
            quadrant = {"X": r + x_0, "Y": c + y_0, "Data": d[r:r + int(d.shape[0] / 2), c:c + int(d.shape[1] / 2)]}
            quadrants.append(quadrant)

    return quadrants


def _otsu_method(t_data):
    bins = int(np.max(t_data)) + 1

    if bins <= 1:
        bins += 1

    hist, bin_centers = histogram(t_data.ravel(), bins, source_range='image')
    hist = hist.astype(float)

    print(len(bin_centers))

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]

    return threshold


def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp


def imshow(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def callback(x):
    pass


def histogram_selection(frame):
    cv.namedWindow('trackbars')

    # create trackbars for color change
    cv.createTrackbar('val', 'trackbars', 0, int(np.max(frame)) * 100, callback)

    counts, values, patches = plt.hist(np.array(frame).flatten(), density=False,
                                       bins=26)
    plt.show()

    while True:
        height, width = frame.shape[:2]
        val = cv.getTrackbarPos('val', 'trackbars') / 100

        mask = cv.inRange(frame, val, np.max(frame))
        dim = (width // 2, height // 2)
        cv.imshow('trackbars', cv.resize(mask, dim, interpolation=cv.INTER_AREA))
        cv.imshow('original image', cv.resize(frame, dim, interpolation=cv.INTER_AREA))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    return val


def _entropy_threhsold(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    print(img.shape)

    H = cv.calcHist([img], [0], None, [255 + 1], [0, 255 + 1])
    H = H / np.sum(H)

    theta = np.zeros(255)
    Hf = np.zeros(255)
    Hb = np.zeros(255)

    for T in range(1, 255):
        Hf[T] = - np.sum(entp(H[:T - 1] / np.sum(H[1:T - 1])))
        Hb[T] = - np.sum(entp(H[T:] / np.sum(H[T:])))
        theta[T] = Hf[T] + Hb[T]

    theta_max = np.argmax(theta)
    img_out = img > theta_max

    plt.plot([i for i in range(255 + 1)], H, 'b')
    plt.plot([theta_max], [H[theta_max]], 'ro')
    plt.plot([theta_max, theta_max], [0, H[theta_max]], 'r')

    plt.annotate("$\\theta_{opt}$",
                 xy=(theta_max, H[theta_max]),
                 xytext=(10 + theta_max, H[theta_max]),
                 color='r')

    plt.annotate("$T_{opt}$",
                 xy=(theta_max, 0),
                 xytext=(10 + theta_max, 0),
                 color='r')

    img_out = img_out.astype(int) * 255

    print(Counter(img_out.flatten()))
    #
    # cv.imshow("asdasd", img_out * 255)
    # cv.waitKey(0)

    return img_out


def reject_outliers(data, m=10.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.

    m = np.percentile(data, 99)
    print(m)

    return data[data < m]


def _histogram_threshold(t_data, std, conf):
    # iqr = np.subtract(*np.percentile(t_data * 100, [75, 25]))
    # n = t_data.shape[0] * t_data.shape[1]
    # h = max(2 * iqr * math.pow(n, -1 * (1.0 / 3.0)), 1)
    # n_bins = max(int((np.max(t_data * 100) - np.min(t_data * 100)) / h), 1)

    print("---------> MAX:", max(np.array(t_data).flatten()))
    print("---------> MIN:", min(np.array(t_data).flatten()))
    # print("---------> n_bins:", n_bins)

    data = reject_outliers(np.array(t_data).flatten())
    # data = np.array(t_data)

    counts, values, patches = plt.hist(data.flatten(), density=False,
                                       bins=conf.n_bins)  # `density=False` would

    pixel_ratio = np.sum(counts[1:]) / np.sum(counts)

    print("% of pixels > first bin", 100 * pixel_ratio, "%")

    # max_val = np.argmax(counts)
    max_val = 0
    middle_val = int((max_val + 1 + len(counts)) / 2)
    val = max(middle_val - int((len(counts) - max_val) * std * conf.noise_scaler), 1)
    print("Val", val)
    print("Middle val", middle_val)
    print("Argmax", max_val)
    # print("Selected threshold: %s" % str(threshold))
    print("Standard Deviation: %s" % str(std))

    val = min(val, len(counts) - 1 - val)
    threshold = values[max_val + val]

    if conf.debug:
        plt.axvline(x=threshold, color="r")
        plt.show()
        plt.clf()

    return threshold


def _tiered_threshold_method(d):
    sub_image_size = 8
    sub_image_mean_values = []

    for r in range(0, d.shape[0], sub_image_size):
        for c in range(0, d.shape[1], sub_image_size):
            mean_val = np.mean(d[r:r + sub_image_size, c:c + sub_image_size])
            sub_image_mean_values.append(mean_val)

    sub_image_mean_values = np.array(sub_image_mean_values)

    xmin = min(sub_image_mean_values)
    xmax = max(sub_image_mean_values)

    # get evenly distributed numbers for X axis.
    x = np.linspace(xmin, xmax, 1000)  # get 1000 points on x axis

    # get actual kernel density.
    density = gaussian_kde(sub_image_mean_values)
    y = density(x)
    y = y / max(y)

    low_val = x[np.where(y > 0.85)][-1]
    med_val = x[np.where(y > 0.3)][-1]
    hi_val = x[np.where(y > 0.0025)][-1]

    return [low_val, med_val, hi_val]


def grid_based_segmentation():
    flattened_marker_images, markers_data, marker_names = get_all_point_data()

    conf = ThresholdConfig()
    conf.dump()

    y_pred = []

    start_time = time.time()

    for point_idx, marker_data in enumerate(markers_data):
        print("Point %s" % str(point_idx + 1))

        marker_dict = dict(zip(marker_names, marker_data))
        data = []

        for marker in config.mask_clusters[config.selected_segmentation_mask_type]:
            data.append(marker_dict[marker])

        data = np.nanmean(np.array(data), axis=0)
        data = gaussian_filter(data, sigma=conf.sigma)

        overall_std = np.std(data)

        if overall_std > conf.max_std:  # Arbitrary threshold
            q = deque()
            q.append({"X": 0, "Y": 0, "Data": data})
            mask = np.zeros((data.shape[0], data.shape[1]), np.uint8)

            if conf.debug:
                grid_image = np.zeros((data.shape[0], data.shape[1]), np.uint8)

            while len(q) > 0:
                d = q.popleft()
                t_data = d["Data"]
                std = np.std(t_data)

                if conf.debug:
                    color_map = plt.imshow(t_data)
                    color_map.set_cmap("viridis")
                    plt.colorbar()

                    plt.show()

                if conf.debug:
                    print("Current Length of Queue: %s" % (str(len(d))))
                    print("[X: %s, Y: %s, Size: %s]" % (str(d["X"]), str(d["Y"]), str(d["Data"].shape)))
                    print("Standard Deviation of Current Block: %s" % std)

                if std > conf.max_std and t_data.shape[0] > conf.min_grid_size:
                    quadrants = _split_image_into_quadrants(t_data, d["X"], d["Y"])
                    for quad in quadrants:
                        q.append(quad)
                else:
                    threshold = _histogram_threshold(t_data, std, conf)

                    m = cv.inRange(t_data, threshold, np.max(t_data))

                    if (float(cv.countNonZero(m)) / float(m.shape[0] * m.shape[1])) > conf.max_area_coverage_threshold:
                        m = ((np.logical_not(m / 255)) * 255).astype(np.uint8)

                    mask[d["X"]:d["X"] + d["Data"].shape[0], d["Y"]:d["Y"] + d["Data"].shape[1]] = m

                    if conf.debug:
                        grid_image[d["X"]:d["X"] + d["Data"].shape[0], d["Y"]:d["Y"] + d["Data"].shape[1]] = m
                        cv.rectangle(grid_image,
                                     (d["Y"], d["X"]),
                                     (d["Y"] + d["Data"].shape[1], d["X"] + d["Data"].shape[0]),
                                     (255, 255, 255),
                                     2)

                    if conf.debug:
                        cv.imshow("Mask", mask)
                        cv.imshow("Grid", grid_image)
                        cv.waitKey(0)
        else:
            if conf.debug:
                color_map = plt.imshow(data)
                color_map.set_cmap("viridis")
                plt.colorbar()
                plt.show()
                plt.clf()

            threshold = _histogram_threshold(data, overall_std, conf)

            # Segmentation
            grid_image = np.zeros((data.shape[0], data.shape[1]), np.uint8)
            mask = cv.inRange(data, threshold, np.max(data))

        if (float(cv.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])) > conf.max_area_coverage_threshold:
            mask = ((np.logical_not(mask / 255)) * 255).astype(np.uint8)

        y_pred.append(mask)

        if conf.debug:
            cv.imshow("Mask", mask)
            cv.imshow("Grid", grid_image)
            cv.waitKey(0)

    end_time = time.time()

    cv.imshow("Grid-Based Thresholding", y_pred[2])
    cv.imshow("DeepCell", flattened_marker_images[2])
    cv.waitKey(0)

    f = open(conf.results_dir + "/results.txt", "w")
    conf.dump_to_file(f)

    f.write("Segmentation complete in %s seconds" % str(end_time - start_time))

    output_dir = "%s/masks" % conf.results_dir
    mkdir_p(output_dir)

    for point_idx, y_pred_img in enumerate(y_pred):
        cv.imwrite(output_dir + "/Point%s.png" % str(point_idx + 1), y_pred_img)

    y_pred = np.expand_dims((np.array(y_pred) / 255).astype(np.uint8), axis=-1)
    y_true = np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in flattened_marker_images])
    y_true = np.expand_dims(np.array(y_true / 255).astype(np.uint8), axis=-1)

    # Average Results
    f.write("\n\nAverage Results\n")
    f.write("=" * 10)
    f.write("\nIOU: %s" % str(mean_iou_np(y_true, y_pred)))
    f.write("\nDice: %s" % str(mean_dice_np(y_true, y_pred)))
    f.write("\nF1 Score: %s" % str(f1_score(y_true.flatten(), y_pred.flatten())))
    f.write("\nPrecision Score: %s" % str(precision_score(y_true.flatten(), y_pred.flatten())))
    f.write("\nRecall Score: %s" % str(recall_score(y_true.flatten(), y_pred.flatten())))
    f.write("\nAccuracy Score: %s" % str(accuracy_score(y_true.flatten(), y_pred.flatten())))

    per_point_results = []

    f.write("\n\nPer-Point Results\n")
    f.write("=" * 10)

    for i in range(len(y_true)):
        f.write("\nPoint %s Results\n" % str(i + 1))
        f.write("=" * 10)
        f1 = f1_score(y_true[i].flatten(), y_pred[i].flatten())
        p = precision_score(y_true[i].flatten(), y_pred[i].flatten())
        r = recall_score(y_true[i].flatten(), y_pred[i].flatten())
        accuracy = accuracy_score(y_true[i].flatten(), y_pred[i].flatten())
        f.write("\nF1 Score: %s" % str(f1))
        f.write("\nPrecision Score: %s" % str(p))
        f.write("\nRecall Score: %s" % str(r))
        f.write("\nAccuracy Score: %s" % str(accuracy))
        per_point_results.append(f1)

    f.write("\nF1 Score variance: %s" % str(np.var(per_point_results)))
    f.close()


def metrics_np(y_true, y_pred, metric_name, metric_type='standard', drop_last=True, mean_per_class=False,
               verbose=False):
    """
    Compute mean metrics of two segmentation masks, via numpy.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape,
                                                                                                       y_pred.shape)
    assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    num_classes = y_pred.shape[-1]
    # if only 1 class, there is no background class and it should never be dropped
    drop_last = drop_last and num_classes > 1

    if not flag_soft:
        if num_classes > 1:
            # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
            y_pred = np.array([np.argmax(y_pred, axis=-1) == i for i in range(num_classes)]).transpose(1, 2, 3, 0)
            y_true = np.array([np.argmax(y_true, axis=-1) == i for i in range(num_classes)]).transpose(1, 2, 3, 0)
        else:
            y_pred = (y_pred > 0).astype(int)
            y_true = (y_true > 0).astype(int)

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)  # or, np.logical_and(y_pred, y_true) for one-hot
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection  # or, np.logical_or(y_pred, y_true) for one-hot

    if verbose:
        print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
        print(intersection, np.sum(np.logical_and(y_pred, y_true), axis=axes), union,
              np.sum(np.logical_or(y_pred, y_true), axis=axes))

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = np.not_equal(union, 0).astype(int)
    # mask = 1 - np.equal(union, 0).astype(int) # True = 1

    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]

    # return mean metrics: remaining axes are (batch, classes)
    # if mean_per_class, average over batch axis only
    # if flag_naive_mean, average over absent classes too
    if mean_per_class:
        if flag_naive_mean:
            return np.mean(metric, axis=0)
        else:
            # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
            return (np.sum(metric * mask, axis=0) + smooth) / (np.sum(mask, axis=0) + smooth)
    else:
        if flag_naive_mean:
            return np.mean(metric)
        else:
            # mean only over non-absent classes
            class_count = np.sum(mask, axis=0)
            return np.mean(np.sum(metric * mask, axis=0)[class_count != 0] / (class_count[class_count != 0]))


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='iou', **kwargs)


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='dice', **kwargs)


if __name__ == '__main__':
    grid_based_segmentation()

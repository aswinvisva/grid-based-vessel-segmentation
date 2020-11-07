import os
import time
from collections import deque
from statistics import stdev

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from skimage.exposure import histogram
from sklearn.metrics import f1_score, accuracy_score

from graph_based_segmentation.main import segment
from utils.markers_feature_gen import preprocess_marker_data, normalize_expression_data
from utils.mibi_reader import get_all_point_data
import config.config_settings as config


def _split_image_into_quadrants(d, x_0, y_0):
    quadrants = []

    for r in range(0, d.shape[0], int(d.shape[0] / 2)):
        for c in range(0, d.shape[1], int(d.shape[1] / 2)):
            quadrant = {"X": r + x_0, "Y": c + y_0, "Data": d[r:r + int(d.shape[0] / 2), c:c + int(d.shape[1] / 2)]}
            quadrants.append(quadrant)

    return quadrants


def _otsu_method(t_data):
    hist, bin_centers = histogram(t_data.ravel(), 256, source_range='image')
    hist = hist.astype(float)

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


def grid_based_segmentation(verbose=False, debug=False):
    flattened_marker_images, markers_data, marker_names = get_all_point_data()

    y_pred = []
    max_std = 0.3
    min_grid_size = 128

    start_time = time.time()

    for marker_data in markers_data:
        marker_dict = dict(zip(marker_names, marker_data))
        data = []

        for marker in config.marker_clusters["Vessels"]:
            data.append(marker_dict[marker])

        data = np.nanmean(np.array(data), axis=0)
        data = gaussian_filter(data, sigma=2)

        std = np.std(data)

        if std > max_std:  # Arbitrary threshold
            q = deque()
            q.append({"X": 0, "Y": 0, "Data": data})
            mask = np.zeros((data.shape[0], data.shape[1]), np.uint8)

            if debug:
                grid_image = np.zeros((data.shape[0], data.shape[1]), np.uint8)

            while len(q) > 0:
                d = q.popleft()
                t_data = d["Data"]
                std = np.std(t_data)

                if debug:
                    color_map = plt.imshow(t_data)
                    color_map.set_cmap("viridis")
                    plt.colorbar()

                    plt.show()

                if verbose:
                    print("Current Length of Queue: %s" % (str(len(d))))
                    print("[X: %s, Y: %s, Size: %s]" % (str(d["X"]), str(d["Y"]), str(d["Data"].shape)))
                    print("Standard Deviation of Current Block: %s" % std)

                if std > max_std and t_data.shape[0] > min_grid_size:
                    quadrants = _split_image_into_quadrants(t_data, d["X"], d["Y"])
                    for quad in quadrants:
                        q.append(quad)
                else:
                    threshold = _otsu_method(t_data)

                    m = cv.inRange(t_data, threshold, np.max(t_data))
                    mask[d["X"]:d["X"] + d["Data"].shape[0], d["Y"]:d["Y"] + d["Data"].shape[1]] = m

                    if debug:
                        grid_image[d["X"]:d["X"] + d["Data"].shape[0], d["Y"]:d["Y"] + d["Data"].shape[1]] = m
                        cv.rectangle(grid_image,
                                     (d["X"], d["Y"]),
                                     (d["X"] + d["Data"].shape[0], d["Y"] + d["Data"].shape[1]),
                                     (255, 255, 255),
                                     2)

                    if debug:
                        cv.imshow("ASD", mask)
                        cv.waitKey(0)
        else:
            # Segmentation
            mask = cv.inRange(data, np.percentile(data, 99), np.max(data))
        y_pred.append(mask)

        if debug:
            cv.imshow("ASD", mask)
            cv.imshow("ADSDS", grid_image)
            cv.waitKey(0)

    end_time = time.time()

    print("Segmentation complete in %s seconds" % str(end_time - start_time))

    y_pred = np.expand_dims((np.array(y_pred) / 255).astype(np.uint8), axis=-1)
    y_true = np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in flattened_marker_images])
    y_true = np.expand_dims(np.array(y_true / 255).astype(np.uint8), axis=-1)

    cv.imshow("Pred", y_pred[20] * 255)
    cv.imshow("True", y_true[20] * 255)
    cv.waitKey(0)

    print("IOU", mean_iou_np(y_true, y_pred))
    print("Dice", mean_dice_np(y_true, y_pred))
    print("F1 Score", f1_score(y_true.flatten(), y_pred.flatten()))
    print("Accuracy Score", accuracy_score(y_true.flatten(), y_pred.flatten()))


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

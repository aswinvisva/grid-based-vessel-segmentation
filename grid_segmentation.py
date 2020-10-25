import os
import sys
sys.path.append("oliveria-lab-ml")

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats as stats

from graph_based_segmentation.main import segment
from utils.markers_feature_gen import preprocess_marker_data, normalize_expression_data
from utils.mibi_reader import get_all_point_data
import config.config_settings as config


def grid_based_segmentation(grid_size=16):
    flattened_marker_images, markers_data, marker_names = get_all_point_data()

    selected_markers = ["SMA", "CD31", "GLUT1", "vWF"]

    test_point = markers_data[0]
    marker_data_split_cells = []

    scaling_factor = config.scaling_factor
    expression_type = config.expression_type
    transformation = config.transformation_type
    normalization = config.normalization_type
    n_markers = config.n_markers

    for r in range(0, test_point.shape[1], grid_size):
        for c in range(0, test_point.shape[2], grid_size):
            test_point_mask = np.zeros((test_point.shape[1], test_point.shape[2]), np.uint8)
            test_point_mask[r:r + grid_size, c:c + grid_size] = 1
            marker_data_split_cells.append(test_point_mask)

    expression_vector = []

    for marker_data_cell_mask in marker_data_split_cells:
        data_vec = []

        for marker in test_point:
            result = cv.bitwise_and(marker, marker, mask=marker_data_cell_mask)

            marker_data = preprocess_marker_data(result,
                                                 marker_data_cell_mask,
                                                 expression_type=expression_type)

            data_vec.append(marker_data)

        expression_vector.append(np.array(data_vec))

    expression_vector = normalize_expression_data(expression_vector,
                                                  transformation=transformation,
                                                  normalization=normalization,
                                                  scaling_factor=scaling_factor,
                                                  n_markers=n_markers)

    expression_img = np.zeros((test_point.shape[1], test_point.shape[2], 3), np.uint8)

    selected_data = []
    for idx, grid_cell_vec in enumerate(expression_vector):
        avg_marker_expression = []
        for marker_idx, marker_name in enumerate(marker_names):
            if marker_name in selected_markers:
                avg_marker_expression.append(grid_cell_vec)

        selected_data.append(np.mean(avg_marker_expression))

    for idx, marker_data_cell_mask in enumerate(marker_data_split_cells):
        color = plt.get_cmap('hot')(selected_data[idx])
        color = (255 * color[0], 255 * color[1], 255 * color[2])

        expression_img[np.where(marker_data_cell_mask == 1)] = color

    plt.imshow(expression_img)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('hot'))
    plt.colorbar(sm)
    plt.show()


if __name__ == '__main__':
    grid_based_segmentation()

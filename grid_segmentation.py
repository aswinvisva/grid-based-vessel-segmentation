import numpy as np

from graph_based_segmentation.main import segment
from utils.mibi_reader import get_all_point_data


# def segment(marker_img, k=3):
#     pass


def grid_based_segmentation():
    flattened_marker_images, markers_data, marker_names = get_all_point_data()

    for img in markers_data:
        sigma = 0.05
        k = 500
        min = 50

        segment(img, sigma, k, min)


if __name__ == '__main__':
    grid_based_segmentation()

import numpy as np


def create_median_mask(disparity_map, valid_depth_mask, rect=None):
    """
    Return a mask selecting the median layer, plus shadows.
    :param disparity_map:
    :param valid_depth_mask:
    :param rect:
    :return:
    """
    if rect is not None:
        x, y, w, h = rect
        disparity_map = disparity_map[y:y + h, x:x + w]
        valid_depth_mask = valid_depth_mask[y:y + h, x:x + w]
    median = np.median(disparity_map)
    return np.where((disparity_map == 0) | (abs(valid_depth_mask - median) < 12), 255, 0).astype(np.uint8)

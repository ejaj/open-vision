import numpy as np
import scipy.interpolate


def create_lookup_array(func, length=256):
    """
    The lookup values are clamped to [0, length - 1].
    :param func:
    :param length:
    :return:
    """
    if func is None:
        return None
    lookup_array = np.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookup_array[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookup_array


def apply_lookup_array(lookup_array, src, dst):
    """
    Map a source to a destination using a lookup.
    :param lookup_array:
    :param src:
    :param dst:
    :return:
    """
    if lookup_array is None:
        return
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = lookup_array[src[i, j]]


def create_curve_func(points):
    """
    Return a function derived from control points.
    :param points:
    :return:
    """
    if points is None:
        return None
    num_points = len(points)
    if num_points < 2:
        return None
    xs, ys = zip(*points)

    if num_points < 3:
        kind = 'linear'
    elif num_points < 4:
        kind = 'quadratic'
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)


def create_composite_func(func0, func1):
    """
    Return a composite of two functions.
    :param func0:
    :param func1:
    :return:
    """
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))

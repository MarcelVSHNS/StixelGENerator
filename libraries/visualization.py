import numpy as np
from waymo_open_dataset.utils import camera_segmentation_utils

import matplotlib.pyplot as plt
import tensorflow as tf


def rgba(r, r_max=50):
    """Generates a color based on range.
  Args:
    r: the range value of a given point.
    r_max:
  Returns:
    The color for a given range
  """
    c = plt.get_cmap('plasma')((r % r_max) / r_max)
    c = list(c)
    c[-1] = 0.7  # alpha
    return c


def plot_image(camera_image):
    """Plot a camera image."""
    minor_ticks = np.arange(0, 1921, 8)
    fig = plt.figure(figsize=(20, 12))
    # if True; print horizontal cols on img
    if False:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(minor_ticks)
        plt.grid(axis='x')
    else:
        plt.grid(False)
    plt.imshow(tf.image.decode_jpeg(camera_image.image))


def plot_points_on_image(projected_points, camera_image, rgba_func, point_size=15.0, objects=[], with_range=True):
    """Plots points on a camera image.
      Args:
        projected_points: [N, 3] numpy array. The inner dims are
          [camera_x, camera_y, range].
        camera_image: jpeg encoded camera image.
        rgba_func: a function that generates a color from a range value.
        objects: ...
        point_size: the point size.
    """
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    if with_range:
        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))
        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    else:
        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
        plt.scatter(xs, ys, c='r', s=point_size * 2, edgecolors="none")

    xo = []
    yo = []

    if len(objects) != 0:
        for point in objects:
            xo.append(point[0])  # width, col
            yo.append(point[1])  # height, row
        plt.scatter(xo, yo, c='r', s=point_size * 2, edgecolors="none")
    plt.show()


def plot_stixel_on_image(stixel_points, camera_image, stixel_height=80, stixel_width=8, thickness=16):
    """
    Plots all found Stixel on image with depth colors
    Args:
        stixel_points: a concatenated list of stixel in shape (col, row, depth)
        camera_image:
        stixel_height: t.b.d.
        stixel_width: t.b.d.
        thickness:
    """
    plot_image(camera_image)

    xs = []
    ys = []
    color = []
    stixel_points = np.asarray(stixel_points)
    depth_vals = stixel_points[..., [2]]
    max_depth = np.amax(depth_vals)

    # gts means groundTruthStixel
    for gts in stixel_points:
        xs.append(gts[0])
        ys.append(gts[1])
        color.append(rgba(gts[2], max_depth))

    plt.scatter(xs, ys, c=color, s=thickness, edgecolors="none")
    plt.show()


def change_reference_point_on_image(projected_points, width=1920, height=1280):
    """
    Short function to change the reference point from upper left to lower left - caused to StixelNet requirements
    http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Levi2015BMVC.pdf
    and
    https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Garnett_Real-Time_Category-Based_and_ICCV_2017_paper.pdf
    :param projected_points:
    :param width:
    :param height:
    :return:
    """
    corrected_points = []
    for point in projected_points:
        corrected_points.append((point[0], height - point[1]))
    return projected_points


def rgba_cut_class(class_num):
    if class_num == 0:
        # blue
        return [0, 0, 1]
    if class_num == 1:
        # green
        return[0, 1, 0]
    # red
    return [1, 0, 0]


def plot_list_of_cuts_on_image(image, list_of_cuts, point_size=10.0):
    """
    Args:
        image: the related image to the cuts
        list_of_cuts: found cuts by the segmentation, expected shape: [..., (col, row, class)]
        point_size: how big the point will be drawn
    Returns:
    """
    plot_image(image)

    xs = []
    ys = []
    colors = []

    for point in list_of_cuts:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_cut_class(point[2]))
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

    plt.show()
    return


def plot_segmented_image(segmentation_labels, instance_labels):
    camera_labels = camera_segmentation_utils.panoptic_label_to_rgb(segmentation_labels, instance_labels)
    plt.figure(figsize=(64, 60))
    plt.imshow(camera_labels)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    return

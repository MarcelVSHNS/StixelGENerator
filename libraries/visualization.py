import numpy as np
import matplotlib.pyplot as plt


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


def plot_points_on_image(images, laser_points, laser_camera_projections, view=0):
    plt.figure(figsize=(20, 12))
    plt.imshow(images[view])

    points_view_norm = np.linalg.norm(laser_points[view], axis=-1, keepdims=True)
    points = np.concatenate([laser_camera_projections[view][..., 1:3], points_view_norm], axis=-1)

    xs = []
    ys = []
    colors = []
    for point in points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba(point[2]))
    plt.scatter(xs, ys, c=colors, s=12.0, edgecolors="none")
    """
    xo = []
    yo = []
    if len(objects) != 0:
        for point in objects:
            xo.append(point[0])  # width, col
            yo.append(point[1])  # height, row
        plt.scatter(xo, yo, c='r', s=point_size * 2, edgecolors="none")
    """
    plt.show()

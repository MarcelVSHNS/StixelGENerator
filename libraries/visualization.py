import math

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


def plot_points_on_image(images, laser_points, stixels=None, reasons=None):
    plt.figure(figsize=(20, 12))
    plt.imshow(images)

    points_view_norm = np.linalg.norm(laser_points[..., :3], axis=-1, keepdims=True)
    points = np.concatenate([laser_points[..., 3:5], points_view_norm], axis=-1)
    xs = []
    ys = []
    colors = []
    for point in points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba(point[2]))
    plt.scatter(xs, ys, c=colors, s=12.0, edgecolors="none")
    if stixels is not None:
        z_stixel = []
        z_stixel_y = []
        r_stixel = []
        r_stixel_y = []
        if reasons is not None:
            for stixel, reason in zip(stixels, reasons):
                if reason == 1:
                    z_stixel.append(stixel[3])
                    z_stixel_y.append(stixel[4])
                if reason == 2:
                    r_stixel.append(stixel[3])
                    r_stixel_y.append(stixel[4])
            plt.scatter(r_stixel, r_stixel_y, c='#00ff00', s=18.0)
        else:
            for stixel in stixels:
                z_stixel.append(stixel[3])
                z_stixel_y.append(stixel[4])
        plt.scatter(z_stixel, z_stixel_y, c='#ff0000', s=18.0)

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


def plot_points_2d_graph(laser_points_by_angle, stixels=None, reasons=None, title="Angle X"):
    """
    Plot a 2D Graph of a single angle of the LiDAR
    Args:
        with_stixel: calculates the stixel for the given points
        title: Name the plot
        laser_points_by_angle: Expected shape is [..., [x,y,z, img_x, img_y]
    """
    rs = []
    zs = []
    laser_points_by_angle_fliped = np.flip(laser_points_by_angle, axis=0)
    for point in laser_points_by_angle_fliped:
        x = point[0]
        y = point[1]
        # take z over r
        zs.append(point[2])
        # pythagoras to receive the radius
        rs.append(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))

    #Calculate slope
    m = []
    # by fist appending a zero, the following point is selected if it hits the threshold
    m.append(0.0)
    for i in range(len(rs)-1):
        # Slope
        delta_x = rs[i+1] - rs[i]
        delta_y = zs[i+1] - zs[i]
        m.append(delta_y / delta_x)

    # Visualize
    plt.figure(figsize=(15, 10))
    plt.plot(rs, zs, markersize=0.5)
    plt.plot(rs, zs, 'bo', markersize=1)
    if stixels is not None:
        z_stixel = []
        z_stixel_y = []
        r_stixel = []
        r_stixel_y = []
        if reasons is not None:
            for stixel, reason in zip(stixels, reasons):
                if reason == 1:
                    z_stixel.append(math.sqrt(math.pow(stixel[0], 2) + math.pow(stixel[1], 2)))
                    z_stixel_y.append(stixel[2])
                if reason == 2:
                    r_stixel.append(math.sqrt(math.pow(stixel[0], 2) + math.pow(stixel[1], 2)))
                    r_stixel_y.append(stixel[2])
            plt.plot(r_stixel, r_stixel_y, 'yo', markersize=3)
        else:
            for stixel in stixels:
                z_stixel.append(stixel[3])
                z_stixel_y.append(stixel[4])
        plt.plot(z_stixel, z_stixel_y, 'ro', markersize=2)

    for x, y, g in zip(rs, zs, m):
        label = "{:.3f}".format(g)
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(8, 4),  # distance from text to points (x,y)
                     size=7,
                     ha='center',
                     rotation=45)
    plt.title(title)
    plt.show()


def plot_angle_distribution(points_per_angle):
    fig = plt.figure()
    plt.title(len(points_per_angle))
    ax = fig.add_axes([0, 0, 1, 1])
    angle = range(len(points_per_angle))
    ax.bar(angle, points_per_angle)
    plt.show()

import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


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


def plot_points_on_image(images, laser_points, y_offset=0, stixels=None, reasons=None, title="Points X"):
    plt.figure(figsize=(20, 12))
    plt.tight_layout()
    plt.imshow(images)

    points_view_norm = np.linalg.norm(laser_points[..., :3], axis=-1, keepdims=True)
    points = np.concatenate([laser_points[..., 3:5], points_view_norm], axis=-1)
    xs = []
    ys = []
    colors = []
    for point in points:
        xs.append(point[0])  # width, col
        ys.append(point[1] - y_offset)  # height, row
        colors.append(rgba(point[2]))
    plt.scatter(xs, ys, c=colors, s=7.0, edgecolors="none")
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
    plt.title(title)
    #plt.savefig("test_okt_23.png", bbox_inches='tight', pad_inches=0.1)
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

def plot_z_over_range(point_lists, colors, labels=None):
    # Aufruf der Funktion mit dem *list Operator
    # plot_z_over_range_multiple_lists(points_list_1, points_list_2, colors=colors, labels=labels)
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    assert len(point_lists) == len(colors); "Num lists and num colors doesn't fit for 2D Plot."
    for points, color, label in zip(point_lists, colors, labels or [None] * len(point_lists)):
        # Berechnen der Distanz (Range) aus x und y
        ranges = np.sqrt(points['x']**2 + points['y']**2)
        # Plotten von z über die berechnete Distanz
        ax.scatter(ranges, points['z'], c=color, label=label)

    plt.xlabel('Range')
    plt.ylabel('Z value')
    plt.title('Plot of Z values over Range for multiple lists')
    plt.legend()
    plt.show()


# Funktion zum Plotten von proj_x und proj_y für mehrere Punkte-Listen auf einem gegebenen Bild
def plot_on_image(image: Image, *point_lists, colors, labels=None, y_offset=-35):
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    ax.imshow(np.array(image))
    assert len(point_lists) == len(colors);
    "Num lists and num colors doesn't fit for 2D Plot."
    # Durchlaufe jede Punkte-Liste und zeichne sie auf das Bild
    for points, color, label in zip(point_lists, colors, labels or [None] * len(point_lists)):
        ax.scatter(points['proj_x'], points['proj_y'] + y_offset, color=color, edgecolor='k', s=20, alpha=0.7, label=label)

    plt.title('Clusterpunkte auf dem Bild')
    plt.axis('off')  # Schalte die Achsen aus, um nur das Bild zu sehen
    if labels:
        plt.legend()
    plt.show()


def extract_points_colors_labels(scanline_obj):
    points_list = []
    colors_list = []
    labels_list = []
    # Calculate the number of objects to generate a color map
    num_objects = len(scanline_obj.objects)
    color_map = plt.get_cmap('jet', num_objects)
    for idx, cluster in enumerate(scanline_obj.objects):
        points_list.append(np.asarray(cluster.points))
        color = color_map(idx / num_objects)  # Normalized index to get a color from the map
        colors_list.append(color)
        labels_list.append(f'Cluster {idx + 1}')
    return points_list, colors_list, labels_list
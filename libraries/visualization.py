from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import List, Tuple, Dict, Optional
from libraries import StixelClass
import cv2
from libraries.Stixel import Stixel

colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
          (0, 0, 255), (255, 0, 255)]
stixel_colors = {
    StixelClass.OBJECT: (96, 96, 96),  # dark grey
    StixelClass.TOP: (150, 150, 150)  # grey
}


def plot_z_over_range(point_lists: List[np.array], colors: List[str], labels: List[str] = None):
    # plot_z_over_range_multiple_lists(points_list_1, points_list_2, colors=colors, labels=labels)
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    assert len(point_lists) == len(colors); "Num lists and num colors doesn't fit for 2D Plot."
    for points, color, label in zip(point_lists, colors, labels or [None] * len(point_lists)):
        ranges = np.sqrt(points['x']**2 + points['y']**2)
        ax.scatter(ranges, points['z'], c=color, label=label)

    plt.xlabel('Range')
    plt.ylabel('Z value')
    plt.title('Plot of Z values over Range for multiple lists')
    plt.legend()
    plt.show()


def plot_on_image(image: Image, *point_lists, colors, labels=None, y_offset=0): #-35
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    ax.imshow(np.array(image))
    assert len(point_lists) == len(colors);
    "Num lists and num colors doesn't fit for 2D Plot."
    for points, color, label in zip(point_lists, colors, labels or [None] * len(point_lists)):
        ax.scatter(points['u'], points['v'] + y_offset, color=color, edgecolor='k', s=20, alpha=0.7, label=label)

    plt.title('Clusterpunkte auf dem Bild')
    plt.axis('off')
    if labels:
        plt.legend()
    plt.show()


def extract_points_colors_labels(scanline_obj):
    points_list = []
    colors_list = []
    labels_list = []
    num_objects = len(scanline_obj.objects)
    color_map = plt.get_cmap('jet', num_objects)
    for idx, cluster in enumerate(scanline_obj.objects):
        points_list.append(np.asarray(cluster.points))
        color = color_map(idx / num_objects)
        colors_list.append(color)
        labels_list.append(f'Cluster {idx + 1}')
    return points_list, colors_list, labels_list


def get_color_from_depth(depth, min_depth, max_depth):
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    color = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(image, stixels: List[Stixel], stixel_width=8, alpha=0.2):
    depths = [stixel.depth for stixel in stixels]
    stixels.sort(key=lambda x: x.depth, reverse=True)
    min_depth, max_depth = min(depths), max(depths)
    for stixel in stixels:
        top_left_x, top_left_y = stixel.column, stixel.top_row
        bottom_left_x, bottom_left_y = stixel.column, stixel.bottom_row
        color = get_color_from_depth(stixel.depth, min_depth, 50)
        bottom_right_x = bottom_left_x + stixel_width
        overlay = image.copy()
        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 1)
    return Image.fromarray(image)


def draw_points_on_image(image, points, y_offset=0, coloring_sem: Optional[Dict[str, Tuple]] = None):
    # distances = [calculate_distance(point) for point in points]
    max_distance = max(points['w'])
    cmap = plt.get_cmap('viridis')
    for point in points:
        if coloring_sem is not None:
            color = coloring_sem[point['sem_seg']]
        else:
            normalized_distance = point['w'] / max_distance
            color = cmap(normalized_distance)[:3]
            color = [int(c * 255) for c in color]
        u, v = point['u'], point['v'] + y_offset
        cv2.circle(image, (u, v), 2, color, -1)
    return Image.fromarray(image)


def draw_clustered_points_on_image(image, cluster_list, depth_coloring=False):
    for i in range(0, len(cluster_list)):
        color = colors[i % len(colors)]
        for point in cluster_list[i]:
            if depth_coloring:
                depth = point['w']
                color = get_color_from_depth(depth, 3, 50)
            u, v = int(point['u']), int(point['v'])
            cv2.circle(image, (u, v), 2, color, -1)
    return Image.fromarray(image)


def draw_obj_points_on_image(image, objects, stixels: List[Stixel] = None, y_offset=0):
    for i, cluster_points in enumerate(objects):
        color = colors[i % len(colors)]
        for point in cluster_points.points:
            u, v = point['u'], point['v'] + y_offset
            cv2.circle(image, (u, v), 3, color, -1)
    if stixels is not None:
        for stixel in stixels:
            u, v = stixel.top_point['u'], stixel.top_point['v']
            color = stixel_colors[stixel.position_class]
            cv2.circle(image, (u, v), 3, color, -1)
    return Image.fromarray(image)


def draw_obj_points_2d(objects, stixels: List[Stixel] = None):
    plt.figure(figsize=(12, 8))
    xs = []
    ys = []
    c = []
    for i, cluster_points in enumerate(objects):
        color = colors[i % len(colors)]
        for point in cluster_points.points:
            xs.append(np.sqrt(point['x']**2 + point['y']**2))
            ys.append(point['z'])
            c.append(tuple(val / 255 for val in color))
    if stixels is not None:
        for stixel in stixels:
            xs.append(np.sqrt(stixel.top_point['x'] ** 2 + stixel.top_point['y'] ** 2))
            ys.append(stixel.top_point['z'])
            color = stixel_colors[stixel.position_class]
            c.append(tuple(val / 255 for val in color))
    plt.scatter(xs, ys, c=c, s=20)
    plt.grid(True)
    plt.xlim([0, max(xs)+0.5])
    plt.show()

""" Bounding Box Visualization from https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_camera_only.ipynb """
def show_camera_image(camera_image, layout, title="FRONT"):
  """Display the given camera image."""
  ax = plt.subplot(*layout)
  plt.imshow(camera_image)
  plt.title(title)
  plt.grid(False)
  plt.axis('off')
  return ax

def draw_2d_box(ax, u, v, color, linewidth=1):
  """Draws 2D bounding boxes as rectangles onto the given axis."""
  rect = patches.Rectangle(
      xy=(u.min(), v.min()),
      width=u.max() - u.min(),
      height=v.max() - v.min(),
      linewidth=linewidth,
      edgecolor=color,
      facecolor=list(color) + [0.1])  # Add alpha for opacity
  ax.add_patch(rect)


def draw_3d_wireframe_box(ax, u, v, color, linewidth=3):
  """Draws 3D wireframe bounding boxes onto the given axis."""
  # List of lines to interconnect. Allows for various forms of connectivity.
  # Four lines each describe bottom face, top face and vertical connectors.
  lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7))

  for (point_idx1, point_idx2) in lines:
    line = plt.Line2D(
        xdata=(int(u[point_idx1]), int(u[point_idx2])),
        ydata=(int(v[point_idx1]), int(v[point_idx2])),
        linewidth=linewidth,
        color=list(color) + [0.5])  # Add alpha for opacity
    ax.add_line(line)


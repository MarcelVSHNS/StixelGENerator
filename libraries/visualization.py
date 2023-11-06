import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_z_over_range(point_lists: List[np.array], colors: List[str], labels: List[str] = None):
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

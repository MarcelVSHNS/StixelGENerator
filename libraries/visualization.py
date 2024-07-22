from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from libraries import StixelClass
import cv2
from libraries import Stixel

colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
          (0, 0, 255), (255, 0, 255)]

stixel_colors = {
    StixelClass.OBJECT: (96, 96, 96),  # dark grey
    StixelClass.TOP: (150, 150, 150)  # grey
}


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
def plot_on_image(image: Image, *point_lists, colors, labels=None, y_offset=0): #-35
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


def calculate_depth(x, y, z):
    # Berechnung der Euklidischen Distanz für die Tiefe
    depth = np.sqrt(x**2 + y**2 + z**2)
    return depth


def get_color_from_depth(depth, min_depth, max_depth):
    # Normalisiere die Tiefe zwischen 0 und 1
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    # Konvertiere den normalisierten Wert in einen Farbwert auf der Jet-Farbkarte
    color = plt.cm.RdYlGn(normalized_depth)[:3]
    return tuple(int(c * 255) for c in color)


def draw_stixels_on_image(image, stixels: List[Stixel], stixel_width=8, alpha=0.1):
    depths = [stixel.depth for stixel in stixels]
    stixels.sort(key=lambda x: x.depth, reverse=True)
    min_depth, max_depth = min(depths), max(depths)-10
    for stixel in stixels:
        top_left_x, top_left_y = stixel.column, stixel.top_row
        bottom_left_x, bottom_left_y = stixel.column, stixel.bottom_row
        color = get_color_from_depth(stixel.depth, 3, 50)
        bottom_right_x = bottom_left_x + stixel_width
        overlay = image.copy()
        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_left_y), color, 2)
    return Image.fromarray(image)


def calculate_distance(point):
    return np.sqrt(point['x']**2 + point['y']**2 + point['z']**2)


def draw_points_on_image(image, points, y_offset=0):
    # Farbdefinitionen
    distances = [calculate_distance(point) for point in points]
    max_distance = max(distances)
    cmap = plt.get_cmap('viridis')
    for point, distance in zip(points, distances):
        # Normalisiere die Entfernung für die Farbkarte
        normalized_distance = distance / max_distance
        # Wandle die normalisierte Entfernung in eine RGBA-Farbe um
        color = cmap(normalized_distance)[:3]  # Konvertiere zu RGB
        color = [int(c * 255) for c in color]  # Skaliere auf 0-255
        # Zeichne den Punkt
        proj_x, proj_y = point['proj_x'], point['proj_y'] + y_offset
        cv2.circle(image, (proj_x, proj_y), 3, color, -1)
    return Image.fromarray(image)


def draw_clustered_points_on_image(image, cluster_list, y_offset=0):
    for i, cluster_points in enumerate(cluster_list):
        color = colors[i % len(colors)]
        for point in cluster_points:
            proj_x, proj_y = point['proj_x'], point['proj_y'] + y_offset
            cv2.circle(image, (proj_x, proj_y), 3, color, -1)
    return Image.fromarray(image)

def draw_obj_points_on_image(image, objects, stixels: List[Stixel] = None, y_offset=0):
    for i, cluster_points in enumerate(objects):
        color = colors[i % len(colors)]
        for point in cluster_points.points:
            proj_x, proj_y = point['proj_x'], point['proj_y'] + y_offset
            cv2.circle(image, (proj_x, proj_y), 3, color, -1)
    if stixels is not None:
        for stixel in stixels:
            # Extrahieren Sie die projizierten Koordinaten
            proj_x, proj_y = stixel.top_point['proj_x'], stixel.top_point['proj_y']
            # Wählen Sie die Farbe basierend auf der PositionClass
            color = stixel_colors[stixel.position_class]
            # Zeichnen eines Punktes/Kreises auf das Bild
            cv2.circle(image, (proj_x, proj_y), 3, color, -1)
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

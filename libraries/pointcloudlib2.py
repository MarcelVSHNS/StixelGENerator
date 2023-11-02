import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def remove_ground(points, distance_threshold=0.18, ransac_n=3, num_iterations=10000):
    """
    Removes the ground from a point cloud and retains the associated projection data.
    This function identifies and removes the points that belong to the ground plane
    using the RANSAC algorithm. It then extracts the projection data from the outlier
    points which are not part of the ground.
    Args:
        points: The input array containing point cloud data and projection data.
                              Expected shape is (N, 5) where N is the number of points.
                              The first three columns are x, y, z coordinates of the point cloud.
                              The last two columns proj_x, proj_y are the projection data.
        distance_threshold: The maximum distance a point can be from the plane model
                                  to be considered an inlier.
        ransac_n: The number of points to sample for estimating the plane.
        num_iterations: The number of iterations to run the RANSAC algorithm.
    Returns:
        combined_data_without_ground: An array of shape (M, 5), where M is the number of
                                                    points not belonging to the ground. The first three
                                                    columns are the x, y, z coordinates, and the last two
                                                    columns are the associated projection data.
    """
    # Convert the input array to a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # Use RANSAC to segment the ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    # Select points that are not part of the ground plane
    pcd_without_ground = pcd.select_by_index(inliers, invert=True)
    # Create a boolean mask for inliers
    inlier_mask = np.ones(len(points), dtype=bool)
    inlier_mask[inliers] = False
    outlier_indices = np.arange(len(points))[inlier_mask]
    # Extract the projection data for the outliers
    projection_data = points[outlier_indices, 3:]
    # Convert the point cloud without ground to a NumPy array
    pcd_without_ground_np = np.asarray(pcd_without_ground.points)
    # Merge the point cloud without ground with the projection data
    combined_data_without_ground = np.hstack((pcd_without_ground_np, projection_data))
    return combined_data_without_ground


def remove_line_of_sight(points):
    return points


def group_points_by_angle(points_with_projection, eps=0.00092, min_samples=1):
    """
    Groups points based on their azimuth angle and returns a list of arrays,
    each containing the points (x, y, z, proj_x, proj_y) of the same angle.

    :param points_with_projection: A numpy array of points (x, y, z, proj_x, proj_y).
    :param eps: The maximum distance between two points to be considered in the same cluster.
    :param min_samples: The number of points required in a neighborhood for a point to be considered a core point.
    :return: A list of numpy arrays, each containing points of an angle cluster.
    """
    # Compute the azimuth angle for each point
    points = points_with_projection[:, :3]
    azimuth_angles = np.arctan2(points[:, 1], points[:, 0])
    # Normalize angles between 0 and 2*pi
    # azimuth_angles = np.mod(azimuth_angles, 2 * np.pi)
    # Perform DBSCAN clustering based on azimuth angles only
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(azimuth_angles.reshape(-1, 1))
    # Obtain cluster labels
    labels = db.labels_

    # Group points based on labels and compute average angles
    unique_labels, unique_indices = np.unique(labels, return_index=True)

    # Skip outliers if present
    valid_indices = unique_labels != -1
    unique_labels = unique_labels[valid_indices]
    unique_indices = unique_indices[valid_indices]

    grouped_points = [points_with_projection[labels == label] for label in unique_labels]
    angle_means = [azimuth_angles[labels == label].mean() for label in unique_labels]

    # Sort clusters and average angles by the angles
    angle_means, grouped_points = zip(*sorted(zip(angle_means, grouped_points)))
    print(f"Number of classes for azimuth: {len(grouped_points)}")
    return list(grouped_points), list(angle_means)


def _euclidean_distance_with_raising_eps():
    pass


def stixel_extraction(sorted_stixel):
    stixel = []
    return stixel


def _determine_stixel(points, cluster_labels, ranges, floor_offset=0):
    # Initialize a dictionary to hold points for each cluster
    clusters = {}
    # Group points by their cluster labels
    for point, label, range_val in zip(points, cluster_labels, ranges):
        if label == -1:
            continue  # Skip outliers
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((point, range_val))
    # Convert clusters to a list and sort each cluster by the range
    for label in clusters:
        clusters[label].sort(key=lambda x: x[1])
    # Sort clusters by their average range
    sorted_clusters = sorted(clusters.items(), key=lambda item: np.mean([point[1] for point in item[1]]))
    stixel = stixel_extraction(sorted_clusters)
    return stixel

def cluster_scanline(scanline_pts, eps=2, min_samples=2, metric='euclidean'):
    """
    Applies DBSCAN clustering to scanline points, grouping points that are close in the
    euclidean sense (or another specified metric) into stixels.

    :param scanline_pts: A numpy array of scanline points (x, y, z).
    :param eps: The maximum distance between two points to be considered as in the same cluster.
    :param min_samples: The minimum number of points in a neighborhood for a point to be considered a core point.
    :param metric: The distance metric to use for DBSCAN.
    :return: A numpy array representing stixels, which are the grouped points.
    """
    # Extract the x, y, and z values
    x_values = scanline_pts[:, 0]
    y_values = scanline_pts[:, 1]
    z_values = scanline_pts[:, 2]
    # Compute the radial distance r
    r_values = np.sqrt(x_values ** 2 + y_values ** 2)
    # Sort points by r for clustering
    sorted_indices = np.argsort(r_values)
    sorted_r = r_values[sorted_indices]
    sorted_z = z_values[sorted_indices]
    sorted_scanline_pts = scanline_pts[sorted_indices]
    # Prepare the data for DBSCAN
    db_data = np.column_stack((sorted_r, sorted_z))
    # Check if enough data points are present for clustering
    if len(db_data) > 1:
        # Apply the DBSCAN clustering algorithm
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(db_data)
        labels = db.labels_
    else:
        # Treat the single point as its own cluster
        labels = np.array([0])
    # Identify stixels by cluster
    stixels = _determine_stixel(sorted_scanline_pts, labels, sorted_r)
    return stixels

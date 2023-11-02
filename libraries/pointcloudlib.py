import numpy as np
import pandas as pd
import yaml
import open3d as o3d
import math
from matplotlib.pyplot import get_cmap
from decimal import *

from libraries.visualization import plot_angle_distribution


with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_stixel_from_laser_data(laser_points_by_view):
    """
    Calculates object cuts (stixel) based on laser points over multiple views
    Args:
        laser_points_by_view: An array of laser points by view with cartesian coordinates + their projection [..., [x,y,z, img_x, img_y]]
    Returns: An Array (views) of stixel found by the laser data
    """
    laser_stixel = []
    points_by_angle = []
    # calculate over all views
    for view_laser_points in laser_points_by_view:
        # 1. laser_points_by_angle
        laser_points_by_angle = __order_points_by_angle(view_laser_points)
        points_by_angle.append(laser_points_by_angle)
        gradient_cuts = []
        # calculate stixel over every column
        # Returns a list where every entry is a list of found stixel ([..., [x,y,z, img_x, img_y]])
        pts = laser_points_by_angle
        #np.savez("anglepoints.npz", *pts)
        for angle_points in laser_points_by_angle:
            stixel_from_col, _ = analyse_lidar_col_for_stixel(angle_points)
            #stixel_from_col = detect_stixel_by_col(angle_points)
            if stixel_from_col is not None:
                gradient_cuts.append(stixel_from_col)
        laser_stixel.append(gradient_cuts)
    return laser_stixel, points_by_angle


def __order_points_by_angle(laser_points, img_width=1920, col_width=8):
    """
    Takes a list of laser points, convert them to spherical, orders them and returns sorted cartesian coordinates
    Args:
        laser_points: A list of laser points with cartesian coordinates + their projection to the related image [..., [x,y,z, img_x, img_y]]
        img_width: maximum cols per image regarding the FoV to calculate the fitting grid
        col_width: respect to the target matrix
        f1: factor to fit the grid, coarse measurement step in rad
        fov: coarse FoV in rad
    Returns: a list where every entry is a list of points ([..., [x,y,z, img_x, img_y]]) of the same angle
    """
    # 1. Convert points to spherical coordinates
    laser_points.tolist()
    # Convert to spherical coordinates
    for cart_pt in laser_points:
        cart_pt[:3] = (__cart2sph(cart_pt[:3]))
    sph_np = np.asarray(laser_points)
    # max_az = np.amax(sph_np[..., [1]])
    # min_az = np.amin(sph_np[..., [1]])
    # determine FoV
    # fov = abs(max_az - min_az)
    # determine angle width
    # Waymo LiDAR characteristics: VFoV: 64 shots, HVoV: 2650 steps, depth 4 (range elonga, ...)
    # grid_step = fov / (img_width / col_width * 2)
    laser_steps = 2650 - 1
    angle_step = 2 * math.pi / laser_steps
    # force azimuth into grid
    for cart_pt in laser_points:
        cart_pt[1] = force_angle_to_grid(cart_pt[1], angle_step)
        # cart_pt[1] = round(cart_pt[1], 3)
    # sort the list by angle (azimuth)
    laser_points = sorted(laser_points, key=lambda x: x[1])
    # 3. append all pts to the corresponding angle (divide the big list into lists with the same angle)
    laser_points_by_angle = []
    current_angle = None
    for pts in laser_points:
        if pts[1] != current_angle:
            current_angle = pts[1]
            laser_points_by_angle.append([pts])
        else:
            laser_points_by_angle[-1].append(pts)
    # Draw a histogram of the point per angle distribution
    if config['explore_data_deep']:
        points_per_angle = []
        for lst in laser_points_by_angle:
            points_per_angle.append(len(lst))
        plot_angle_distribution(points_per_angle)
    # 4. Calculate back to cartesian coordinates and convert to numpy
    laser_points_angle_listed = []
    for angle_lst in laser_points_by_angle:
        for sph_pt in angle_lst:
            sph_pt[:3] = __sph2cart(sph_pt[:3])
        laser_points_angle_listed.append(np.array(angle_lst))
    return laser_points_angle_listed


def force_angle_to_grid(angle, step):
    angle_norm = 0
    angle = Decimal(str(round(float(angle), 6)))
    step = Decimal(str(round(float(step), 6)))
    rest = angle % step
    if rest > step / 2:
        angle_norm = angle + (step - rest)
    else:
        angle_norm = angle - rest
    test = angle_norm % step
    assert angle_norm % step == 0
    return angle_norm


def __sph2cart(sph_coord):
    """calculation of spherical coordinates into cartesian coordinates .
        Returns: three values in form [x, y, z]
    """
    r, az, el = sph_coord
    r_cos_theta = r * np.cos(el)
    x = r_cos_theta * np.cos(az)
    y = r_cos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def __cart2sph(cart_coord):
    """calculation of cartesian coordinates into spherical coordinates.
    Args:
        cart_coord: three values in shape [x, y, z]
    Returns: three values in shape [r, az, el]
    """
    x, y, z = cart_coord
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el


def detect_objects_in_point_cloud_numerical(point_cloud, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # Apply RANSAC Algo
    # 'distance_threshold' ist der maximale Abstand eines Punktes von der Ebene, um als Inlier betrachtet zu werden.
    # too small and some ground_pts are not detected, too high and some object_pts count as ground
    # 'ransac_n' ist die Anzahl der Punkte, die für die Ebene verwendet werden.
    # 'num_iterations' ist die Anzahl der Iterationen für RANSAC.
    o3d.visualization.draw_geometries([pcd])
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.24, ransac_n=3, num_iterations=1000)
    # pcd = pcd.select_by_index(inliers, invert=True)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.4, min_points=10, print_progress=True))

    if visualize:
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])
        """
        # Select a cluster by number
        for i in [38]:
            object_num_x = np.where(labels == i)
            pcd_object_x = pcd_without_ground.select_by_index(object_num_x[0].tolist())
            print(i)
            o3d.visualization.draw_geometries([pcd_object_x])
        """


def analyse_lidar_col_for_stixel(laser_points_by_angle, threshold_z_slope=0.125, threshold_z_abs=0.01, threshold_r=0.008, threshold_jump=0.4,
                                 threshold_distance=75.0, investigate=False):
    """
    A function which is simply calculating the slope between each poit pair and applying thresholds to filter for stixel
    Args:
        threshold_z_abs:
        investigate: adds additional prints
        laser_points_by_angle: Expected shape is [..., [x,y,z, img_x, img_y]
        threshold_z_slope: means the detection of an object (z-axle)
        threshold_r: means the detection of a gap (radius)
        threshold_jump: means, the additional radius to delete in case of a back step
        threshold_distance: limits the maximum detectable stixel in m
    Returns:

    """
    # Define
    xs_radius = []
    zs_level = []   # TODO: use laser_points_.. directly
    m = []          # list of slopes from point to point
    delta_r = []    # list of radius differences between points
    delta_z = []

    laser_points_by_angle = np.flip(laser_points_by_angle, axis=0)
    for point in laser_points_by_angle:
        x = point[0]
        y = point[1]
        # take z over r
        zs_level.append(point[2])
        # pythagoras to receive the radius
        xs_radius.append(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))

    # by fist appending a zero, the following point is selected if it hits the threshold
    init_x = xs_radius[0]
    init_y = zs_level[0]
    delta_z.append(0.0)
    if init_x != 0:
        m.append(init_y / init_x)
    else:
        m.append(0.0)
    for i in range(len(xs_radius) - 1):
        # Slope
        delta_x = xs_radius[i + 1] - xs_radius[i]
        if delta_x == 0:
            delta_x = 0.001
        delta_y = zs_level[i + 1] - zs_level[i]
        delta_z.append(delta_y)
        assert delta_x != 0, f"divided by 0"
        m.append(delta_y / delta_x)
        # Viewing gap
        delta_r.append(delta_x)
    # afterward appending, means take the first point in case of hitting the threshold
    delta_r.append(0.0)
    stixel_lst = []
    if investigate:
        print("No. \t r \t m")
    for i in range(len(xs_radius)):
        if investigate:
            print(f"{i}. {xs_radius[i]} - {m[i]} - {delta_z[i]}")

        # limits the sensor distance
        if xs_radius[i] <= threshold_distance:
            # Evaluates the slope
            if abs(m[i]) >= threshold_z_slope and delta_z[i] >= threshold_z_abs:
                # delete the points before the new one - threshold
                if m[i] <= 0:
                    # keeps all points which are smaller than the current r - threshold
                    for stixel in stixel_lst:
                        if stixel[0] > xs_radius[i] - threshold_jump:
                            stixel[1] = 0
                stixel_lst.append([xs_radius[i], 1])
            else:
                # TODO: experimental: Evaluate the gap
                gap = threshold_r * math.pow(xs_radius[i], 2)
                if delta_r[i] >= gap:
                    if investigate:
                        print(f"with gap: {delta_r[i]} and Tol:{gap}")
                    # Change 0 to 2, to activate
                    stixel_lst.append([xs_radius[i], 0])
                else:
                    stixel_lst.append([xs_radius[i], 0])
        else:
            stixel_lst.append([xs_radius[i], 0])

    stixel_table = np.hstack((laser_points_by_angle, np.asarray(stixel_lst)[..., [1]]))
    # filter by detected stixel or not
    stixels = list(filter(lambda num: num[5] > 0, stixel_table))
    if len(stixels) > 0:
        return np.asarray(stixels)[..., :5], np.asarray(stixels)[..., 5:]
    else:
        return None, None


def force_stixel_into_image_grid(stixel_by_view, grid_step=8):
    """
    Forces all given stixel into the output grid.
    Args:
        stixel_by_view:
        grid_step:
    Returns: a list of views with grid stixel
    """
    grid_stixel_by_view = []
    for view_stixel in stixel_by_view:
        stacked_stixel = np.vstack(view_stixel)
        for stixel in stacked_stixel:
            stixel[3] = normalize_into_grid(stixel[3], step=grid_step)
            if stixel[3] == 1920:
                stixel[3] = 1912
            stixel[4] = normalize_into_grid(stixel[4], step=grid_step)
            if stixel[4] == 1280:
                stixel[4] = 1272
        grid_stixel_by_view.append(stacked_stixel)
    return grid_stixel_by_view


def normalize_into_grid(pos, step=8, offset=0):
    val_norm = 0
    rest = pos % step
    if rest > step / 2:
        val_norm = pos + (step - rest)
    else:
        val_norm = pos - rest
    assert val_norm % step == 0
    return val_norm + offset


def remove_ground(points, distance_threshold=0.18, ransac_n=3, num_iterations=10000):
    """
    Removes the ground from a point cloud and retains the associated projection data.
    This function identifies and removes the points that belong to the ground plane
    using the RANSAC algorithm. It then extracts the projection data from the outlier
    points which are not part of the ground.

    Args:
        points (numpy.ndarray): The input array containing point cloud data and projection data.
                              Expected shape is (N, 5) where N is the number of points.
                              The first three columns are x, y, z coordinates of the point cloud.
                              The last two columns proj_x, proj_y are the projection data.
        distance_threshold (float): The maximum distance a point can be from the plane model
                                  to be considered an inlier.
        ransac_n (int): The number of points to sample for estimating the plane.
        num_iterations (int): The number of iterations to run the RANSAC algorithm.
    Returns:
        combined_data_without_ground (numpy.ndarray): An array of shape (M, 5), where M is the number of
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

def detect_stixel_by_col(points_by_angle):
    pass

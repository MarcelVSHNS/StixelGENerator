import numpy as np
import pandas as pd
import yaml
import open3d as o3d
import math
from matplotlib.pyplot import get_cmap

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
        for angle_points in laser_points_by_angle:
            stixel_from_col, _ = analyse_lidar_col_for_stixel(angle_points)
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
    max_az = np.amax(sph_np[..., [1]])
    min_az = np.amin(sph_np[..., [1]])
    # determine FoV
    fov = abs(max_az - min_az)
    # determine angle width
    grid_step = fov / (img_width / col_width * 2)
    # force azimuth into grid
    for cart_pt in laser_points:
        cart_pt[1] = cart_pt[1] - (cart_pt[1] % grid_step)
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


def analyse_lidar_col_for_stixel(laser_points_by_angle, threshold_z=0.125, threshold_r=0.008, threshold_jump=0.4,
                                 threshold_distance=50.0, investigate=False):
    """
    A function which is simply calculating the slope between each poit pair and applying thresholds to filter for stixel
    Args:
        investigate: adds additional prints
        laser_points_by_angle: Expected shape is [..., [x,y,z, img_x, img_y]
        threshold_z: means the detection of an object (z-axle)
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
    m.append(init_y / init_x)
    for i in range(len(xs_radius) - 1):
        # Slope
        delta_x = xs_radius[i + 1] - xs_radius[i]
        if delta_x == 0:
            delta_x = 0.001
        delta_y = zs_level[i + 1] - zs_level[i]
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
            print(f"{i}. {xs_radius[i]} - {m[i]}")

        # limits the sensor distance
        if xs_radius[i] <= threshold_distance:
            # Evaluates the slope
            if abs(m[i]) >= threshold_z:
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

from dataloader import WaymoDataLoader as Dataset
from libraries import *
import open3d as o3d
import numpy as np
import yaml
import random
from libraries.Stixel import point_dtype_ext

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
if config['dataset'] == "waymo":
    from dataloader import WaymoDataLoader as Dataset, WaymoData as Data
elif config['dataset'] == "kitti":
    from dataloader import KittiDataLoader as Dataset, KittiData as Data
elif config['dataset'] == "kitti-360":
    from dataloader import Kitti360DataLoader as Dataset, Kitti360Data as Data
else:
    raise ValueError("Dataset not supported")


def main():
    """ 0.1 Use the plc-config file to make adjustments and customizations to your data. """
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'],
                               phase=config['phase'],
                               first_only=True)
    if config['exploring']['idx'] is not None:
        assert config["exploring"]["idx"] <= len(dataset)
        idx = config['exploring']['idx']
    else:
        idx = random.randint(0, len(dataset) - 1)
        print(f"Random index: {idx}")
    drive = dataset[idx]
    assert config["exploring"]["frame_num"] <= len(drive), "Check 'first_only' option of the dataloader."
    sample: Data = drive[config["exploring"]["frame_num"]]
    # TODO: explore 10203656353524179475_7625_000_7645_000_25_FRONT

    """ 0.2 Check out if your own camera-lidar projection works (camera calib data are not always and for everyone
     unique explained). It is necessary to calculate the correct bottom point of a finished Stixel. 
     coloring_sem=waymo_laser_label_color """
    # new_pts = sample.projection_test()
    # points_on_img = draw_points_on_image(np.array(sample.image), sample.points)
    # points_on_img.show()
    # alt_non_gnd = segment_ground(sample.all_points, sample.mask, sample.laser_projection_points)
    # points_on_img = draw_points_on_image(np.array(sample.image), alt_non_gnd)
    # points_on_img.show()

    """ Semantic filtering """
    #pts_filter_sem_seg = filter_points_by_semantic(points=sample.points,
    #                                               param=dataset.config['semantic_filter'])
    #points_on_img = draw_points_on_image(np.array(sample.image), pts_filter_sem_seg, coloring_sem=waymo_laser_label_color)
    #points_on_img.show()

    """ 1. Adjust the ground detection. Try to make it rough! the street should disappear every time! Repeat the same
     configuration (libraries/pcl-config.yaml) multiple times to proof your values. Try not to make it precise, the
     bottom point will be recalculated! higher values (distance_threshold) will make the road 'thicker'. The z_max 
     value masks all values above this value to avoid detecting large walls. """
    angled_pts = group_points_by_angle(points=sample.points, param=dataset.config['group_angle'],
                                       camera_info=sample.camera_info)
    # angled_img = draw_points_on_image(np.array(sample.image), angled_pts, color_by_angle=True)
    # angled_img.show()
    #lp_without_ground = sample.points
    lp_without_ground, ground_model = remove_ground(points=angled_pts, param=dataset.config['rm_gnd'])
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_ground)
    # points_on_img.show()

    """ Label filtering """
    pts_filter_bbox, bbox_ids = filter_points_by_label(points=lp_without_ground, bboxes=sample.laser_labels)
    # points_on_img = draw_points_on_image(np.array(sample.image), pts_filter_bbox)
    # points_on_img.show()

    # 1.1 Self explained: adjust the maximum distance of points, be aware that possibly no points exist in the scene
    # if this value is too low
    pts_filter_bbox = remove_far_points(points=pts_filter_bbox, param=dataset.config['rm_far_pts'])
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_far_pts)
    # points_on_img.show()

    # 1.2 Measurement point below the road, do they exist??? ... not anymore ;)
    # lp_without_ground = remove_pts_below_plane_model(points=lp_without_ground, plane_model=ground_model)
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_ground)
    # points_on_img.show()

    """ 2. Depending on your vehicle setup (extrinsic calibration from lidar to camera), the lidar might see areas, 
    which the camera does not - normally because of the lifted viewing angle of the top lidar. This function helps to 
    just use points whose are in the field of view of the camera. Refer to 
    https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Hidden-point-removal
    lp_without_los = remove_line_of_sight(points=lp_plane_model_corrected,
                                          camera_pose=sample.camera_info.extrinsic.xyz,
                                          param=dataset.config['rm_los'])
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_los)
    # points_on_img.show()"""

    """ 3. Check angle grouping. This is important because the algorithm scans for stixel line by line (lidar angle). 
    The visualization shows groups in different colors. You should set the eps to a value which divide every close
    point into one group; more far away points can be clustered with more than one scanline but in general there should
    be scan lines in direction of elevation for every angle."""
    # angled_pts = group_points_by_angle(points=pts_filter_bbox, param=dataset.config['group_angle'], camera_info=sample.camera_info)
    # angled_img = draw_points_on_image(np.array(sample.image), angled_pts, color_by_angle=True)
    # angled_img.show()
    """ explore deep 
    if config['explore_data_deep']:
        print(f'From right to left with {len(angled_pts)} angeles.')
        if len(angled_pts[config['exploring']['col']]) != 0:
            column = Scanline(angled_pts[config['exploring']['col']],
                              camera_info=sample.camera_info,
                              plane_model=ground_model,
                              image_size=dataset.img_size,
                              stixel_width=8,
                              param=dataset.config['stixel_cluster'])
            objects = column.objects
            stixel = column.get_stixels()
            obj_scanline_on_img = draw_obj_points_on_image(np.array(sample.image), objects=objects, stixels=stixel)
            obj_scanline_on_img.show()
            draw_obj_points_2d(objects, stixel)
            print("ende")
        else:
            print(f"No points in that col({config['exploring']['col']}) found.")"""

    """ 4. Check the result or go into details. You should receive a nice looking, coloured representation of stixels. 
    use scanline_cluster_obj params to adjust the vertical clustering (..._factor). x_threshold is the minimum distance 
    of a top point to be part of the current object or create a new stixel. """
    stixel_gen = StixelGenerator(camera_info=sample.camera_info,
                                 img_size=dataset.img_size,
                                 stixel_width=8,
                                 stixel_param=dataset.config['stixel_cluster'],
                                 angle_param=dataset.config['group_angle'])
    stixel_list = []
    for id in bbox_ids:
        for bbox in sample.laser_labels:
            if bbox.id == id:
                plane = calculate_plane_from_bbox(bbox)
                stixel_gen.plane_model = plane
                bbox_points = pts_filter_bbox[pts_filter_bbox['id'] == id]
                # points_on_img = draw_points_on_image(np.array(sample.image), bbox_points, color_by_angle=True)
                # points_on_img.show()
                stixel_list.append(stixel_gen.generate_stixel(bbox_points))
    stixel = [item for sublist in stixel_list for item in sublist]
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image), stixel, stixel_width=config['grid_step'], draw_grid=True)
    gt_stixel_img.show()

    """
    stx_pts = []
    for stx in stixel_list:
        stx_pts.append(stx.top_point)

    # new_img_pts = sample.projection_test()
    new_pts = sample.inverse_projection(sample.points)
    coord = np.vstack((sample.points['x'], sample.points['y'], sample.points['z'])).T
    coord = np.insert(coord[:, 0:3], 3, 1, axis=1).T
    waymo_cam_RT = np.array([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
    dist_1 = sample.camera_info.T @ waymo_cam_RT @ coord
    dist_1 = dist_1.T
    dist = np.linalg.norm(dist_1, axis=1)
    # dist = np.sqrt(new_img_pts['x']**2 + new_img_pts['y']**2 + new_img_pts['z']**2)
    ws = sample.points['w']
    k = dist/ws

    pcd = o3d.geometry.PointCloud()
    stx_pts = np.array([tuple(row) for row in np.array(stx_pts)], dtype=point_dtype_ext)
    xyz = np.vstack((stx_pts['x'], stx_pts['y'], stx_pts['z'])).T
    xyz_new = np.vstack((sample.points['x'], sample.points['y'], sample.points['z'])).T
    pcd.points = o3d.utility.Vector3dVector(new_pts)

    bounding_boxes = []
    for box in sample.laser_labels:
        center_x, center_y, center_z = box.camera_synced_box.center_x, box.camera_synced_box.center_y, box.camera_synced_box.center_z
        length, width, height = box.camera_synced_box.length, box.camera_synced_box.width, box.camera_synced_box.height
        heading = box.camera_synced_box.heading

        box_corners = np.array([
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2]
        ])

        # Schritt 2: Rotation (Heading) um die Z-Achse anwenden
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        # Eckpunkte drehen und anschließend um das Zentrum verschieben
        rotated_corners = box_corners @ rotation_matrix.T
        rotated_corners += np.array([center_x, center_y, center_z])

        # Schritt 3: Bounding Box in Open3D visualisieren
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Untere Ebene
            [4, 5], [5, 6], [6, 7], [7, 4],  # Obere Ebene
            [0, 4], [1, 5], [2, 6], [3, 7]  # Verbindungen zwischen Ebenen
        ]

        colors = [[1, 0, 0] for i in range(len(lines))]  # Farbe Rot

        # Erstelle LineSet für die Box-Visualisierung
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bounding_boxes.append(line_set)

    # Visualise point cloud
    o3d.visualization.draw_geometries([pcd] + bounding_boxes)
    """


if __name__ == "__main__":
    main()

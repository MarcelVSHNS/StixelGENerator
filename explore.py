import numpy as np
import open3d as o3d
import yaml
import random

from dataloader.WaymoDataset import WaymoDataLoader
from dataloader.AmeiseDataset import AmeiseDataLoader
from libraries.visualization import plot_points_on_image
from libraries.pointcloudlib import get_stixel_from_laser_data
from libraries.pointcloudlib import detect_objects_in_point_cloud_numerical
from libraries.visualization import plot_points_2d_graph
from libraries.pointcloudlib import analyse_lidar_col_for_stixel
from libraries.pointcloudlib import force_stixel_into_image_grid


def main():
    # open Config
    with open('config.yaml') as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    data_dir = config['raw_data_path'] + config['phase']
    dataset_to_use = config['selected_dataset']

    """ load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
    .images (5 views) and .laser_points (top lidar, divided into 5 fitting views). Like e.g.:
        798 tfrecord-files (selected by "idx")
            ~20 Frames (batch size/ dataset - selected by "frame_num")
                5 .images (camera view - selected by index[])
                5 .laser_points (shape of [..., [x, y, z, img_x, img_y]])"""
    if dataset_to_use == "ameise":
        dataset = AmeiseDataLoader(data_dir=data_dir, first_only=True)
    else:
        dataset = WaymoDataLoader(data_dir=data_dir, camera_segmentation_only=False, first_only=True)

    # Example to display the lidar camera projection and provide an exploring data sample
    if config['exploring']['random_idx']:
        idx = random.randint(0, len(dataset))
    else:
        idx = config['exploring']['idx']
    frame_num = config['exploring']['frame_num']
    view = config['exploring']['view']
    col = config['exploring']['col']

    assert len(dataset) >= idx, f'The index is too high, records found: {len(dataset)}'
    #assert len(dataset[idx][frame_num].cameras) != 0, 'The chosen index has no segmentation data'
    sample = dataset[idx][frame_num]

    # Show the Objects by point cloud
    # detect_objects_in_point_cloud_numerical(sample.laser_points[view][..., :3], visualize=True)

    # Basic concept is to load general data like waymo and apply the library functions to the provided data
    # Get Stixel from Laser
    # Get Laser points sorted by angle
    laser_stixel, laser_by_angle = get_stixel_from_laser_data(laser_points_by_view=[sample.image_points[config['exploring']['view']]])

    # Search for Stixel on image
    #stixels, _reasons = analyse_lidar_col_for_stixel([laser_by_angle[-1][col]], investigate=True)
    # Full Point Cloud

    plot_points_on_image(images=sample.cameras[view],
                         laser_points=sample.image_points[view],
                         title=f"Idx = {idx}, Frame: {frame_num}, View: {view}")
    """
    # One angle with stixel
    plot_points_on_image(images=sample.images[view],
                         laser_points=laser_by_angle[view][col],
                         stixels=stixels,
                         reasons=_reasons,
                         title=f"Idx = {idx}, Frame: {frame_num}, View: {view}, Col: {col}")
    # One angle 2D Plot with stixel
    plot_points_2d_graph(laser_points_by_angle=laser_by_angle[view][col],
                         stixels=stixels,
                         reasons=_reasons,
                         title=f"Idx = {idx}, Frame: {frame_num}, View: {view}, Col: {col}")
    """
    # Training data visual
    training_data = force_stixel_into_image_grid(laser_stixel)
    plot_points_on_image(images=sample.cameras[view],
                         laser_points=training_data[-1],
                         title=f"Idx = {idx}, Frame: {frame_num}, View: {view}")
    # Show the Objects by point cloud
    # detect_objects_in_point_cloud_numerical(training_data[view][..., :3], visualize=True)


if __name__ == "__main__":
    main()

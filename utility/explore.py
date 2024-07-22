from dataloader import WaymoDataLoader as Dataset
from libraries import *
import numpy as np
import yaml
import random

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
frame_num = config['exploring']['frame_num']


def main():
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=config['phase'], first_only=True)
    if config['exploring']['idx'] is not None:
        assert config["exploring"]["idx"] <= len(dataset)
        idx = config['exploring']['idx']
    else:
        idx = random.randint(0, len(dataset) - 1)
    drive = dataset[idx]
    assert config["exploring"]["frame_num"] <= len(drive)
    sample = drive[config["exploring"]["frame_num"]]

    """ 0. Check out if your own camera-lidar projection works (camera calib data are not always and for everyone
     unique explained). It is necessary to calculate the correct bottom point of a finished Stixel."""
    # new_pts = sample.projection_test()
    # points_on_img = draw_points_on_image(np.array(sample.image), new_pts)
    # points_on_img.show()

    """ 1. Adjust the ground detection. Try to make it rough! the street should disappear every time! Repeat the same
     configuration (libraries/pcl-config.yaml) multiple times to proof your values. Try not to make it precise, the
     bottom point will be recalculated! higher values (distance_threshold) will make the road 'thicker'. The z_max 
     value masks all values above this value to avoid detecting large walls. """
    lp_without_ground, ground_model = remove_ground(sample.points)
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_ground)
    # points_on_img.show()

    # Self explained: adjust the maximum distance of points
    lp_without_far_pts = remove_far_points(lp_without_ground)
    # points_on_img = draw_points_on_image(np.array(sample.image), lp_without_far_pts)
    # points_on_img.show()

    # Measurement point below the road, do they exist??? ... not anymore ;)
    lp_plane_model_corrected = remove_pts_below_plane_model(lp_without_far_pts, ground_model)
    points_on_img = draw_points_on_image(np.array(sample.image), lp_plane_model_corrected)
    points_on_img.show()

    """ 2. """
    lp_without_los = remove_line_of_sight(lp_plane_model_corrected, sample.camera_info.extrinsic.xyz)
    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_los)
    points_on_img.show()

    """ 3. Check angle grouping """
    # angled_pts = group_points_by_angle(lp_without_los)
    # angled_img = draw_clustered_points_on_image(np.array(sample.image), angled_pts)
    # angled_img.show()

    stixel_gen = StixelGenerator(camera_info=sample.camera_info, img_size=dataset.img_size,
                                 plane_model=ground_model, stixel_width=8)
    stixel_list = stixel_gen.generate_stixel(lp_without_los)
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image), stixel_list, stixel_width=config['grid_step'])
    gt_stixel_img.show()


if __name__ == "__main__":
    main()

from dataloader import KittiDataLoader as Dataset
from libraries import *
import numpy as np
import yaml

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
frame_num = config['exploring']['frame_num']


def main():
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=config['phase'], first_only=True)
    assert config["exploring"]["idx"] <= len(dataset)
    drive = dataset[config["exploring"]["idx"]]
    assert config["exploring"]["frame_num"] <= len(drive)
    sample = drive[config["exploring"]["frame_num"]]

    # lp means lidar_points
    lp_without_ground, ground_model = remove_ground(sample.points)
    lp_without_far_pts = remove_far_points(lp_without_ground)
    lp_plane_model_corrected = remove_pts_below_plane_model(lp_without_far_pts, ground_model)
    lp_without_los = remove_line_of_sight(lp_plane_model_corrected, sample.camera_pov)
    stixel_gen = StixelGenerator(camera_info=sample.camera_info, img_size=dataset.img_size,
                                 plane_model=ground_model)
    stixel_list = stixel_gen.generate_stixel(lp_without_los)

    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_los)
    points_on_img.show()
    """
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image), stixel_list, stixel_width=config['grid_step'])
    gt_stixel_img.show()
    """


if __name__ == "__main__":
    main()

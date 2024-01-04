import yaml
from dataloader.KittiDataset import KittiDataLoader as Dataset, KittiData as Data
from libraries.visualization import draw_points_on_image, draw_stixels_on_image
from libraries.pointcloudlib import *
import numpy as np


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=config['phase'], first_only=True)
    scene: Data = dataset[0]
    sample = scene[0]

    lp_without_ground, ground_model = remove_ground(sample.points)

    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_ground)
    points_on_img.show()

    lp_without_far_pts = remove_far_points(lp_without_ground)
    lp_plane_model_corrected = remove_pts_below_plane_model(lp_without_far_pts, ground_model)
    lp_without_los = remove_line_of_sight(lp_plane_model_corrected, sample.camera_info.extrinsic.xyz)
    stixel_gen = StixelGenerator(camera_info=sample.camera_info, img_size=dataset.img_size,
                                 plane_model=ground_model)
    stixel_list = stixel_gen.generate_stixel(lp_without_los)
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image), stixel_list)
    gt_stixel_img.show()

    print(scene.name)


if __name__ == "__main__":
    main()

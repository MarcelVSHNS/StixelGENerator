from dataloader.KittiDataset import KittiDataLoader as Dataset
from libraries.pointcloudlib import *
from libraries.visualization import draw_stixels_on_image, draw_points_on_image, draw_clustered_points_on_image, draw_obj_points_on_image, draw_obj_points_2d
import yaml


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
frame_num = config['exploring']['frame_num']


def main():
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=config['phase'], first_only=True)
    set = dataset[0]
    sample = set[-1]
    # 105[-1] cyclist
    """
    ground = []
    for frame in dataset[0]:
        lp_without_ground, ground_height = remove_ground(frame.points)
        ground.append(ground_height)
        print(ground_height)
    durchschnitt = sum(ground) / len(ground)
    print(f"mean: {durchschnitt}")
    """
    lp_without_ground, ground_model = remove_ground(sample.points)
    lp_without_far_pts = remove_far_points(lp_without_ground)
    lp_plane_model_corrected = remove_pts_below_plane_model(lp_without_far_pts, ground_model)
    lp_without_los = remove_line_of_sight(lp_plane_model_corrected, sample.camera_pov)
    stixel_gen = StixelGenerator(camera_info=sample.camera_info, img_size=dataset.img_size,
                                 plane_model=ground_model)
    stixel_list = stixel_gen.generate_stixel(lp_without_los)
    """
    # draw points
    lp_without_ground, ground_height = remove_ground(sample.points)
    lp_without_far_pts = remove_far_points(lp_without_ground)
    lp_without_los = remove_line_of_sight(lp_without_far_pts, sample.camera_pov)
    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_los)
    points_on_img.show()
    
    points_on_img = draw_points_on_image(np.array(sample.image), sample.points)
    points_on_img.show()
    """
    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_ground)
    points_on_img.show()
    """
    points_on_img = draw_points_on_image(np.array(sample.image), lp_without_los)
    points_on_img.show()
    # draw angle
    #angle_on_img = draw_clustered_points_on_image(np.array(sample.image), lp_wg_ordered_by_angle)
    #angle_on_img.show()

    # draw stixel
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image), stixel_list)
    gt_stixel_img.show()
    lp_wg_ordered_by_angle = group_points_by_angle(lp_without_los)
    # single scanline
    if config['explore_data']:
        column = Scanline(lp_wg_ordered_by_angle[131], camera_info=sample.camera_info, image_size=dataset.img_size,
                                 plane_model=ground_model)
        obj_scanline_on_img = draw_obj_points_on_image(np.array(sample.image), column.objects, column.get_stixels())
        obj_scanline_on_img.show()
        draw_obj_points_2d(column.objects, column.get_stixels())
    print(len(stixel_list))
    """

if __name__ == "__main__":
    main()

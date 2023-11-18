from dataloader.AmeiseDataset import AmeiseDataLoader as Dataset
from libraries.pointcloudlib import *
from libraries.visualization import draw_stixels_on_image, draw_points_on_image, draw_clustered_points_on_image, draw_obj_points_on_image, draw_obj_points_2d
import yaml


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
frame_num = config['exploring']['frame_num']

def main():
    dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=config['phase'], first_only=True)
    sample = dataset[40][-1]

    lp_without_ground = remove_ground(sample.points)
    lp_without_far_pts = remove_far_points(lp_without_ground)
    lp_without_los = remove_line_of_sight(lp_without_far_pts, sample.camera_pov)
    lp_wg_ordered_by_angle = group_points_by_angle(lp_without_los)
    stixel: List[List[Stixel]] = []
    for laser_points_by_angle in lp_wg_ordered_by_angle:
        column: Scanline = Scanline(laser_points_by_angle)
        stixel.append(column.get_stixels())  # calculate stixel
    grid_stixel: List[Stixel] = force_stixel_into_image_grid(stixel, dataset.img_size)
    # draw points
    points_on_img = draw_points_on_image(np.array(sample.image), sample.points)
    points_on_img.show()

    # draw angle
    angle_on_img = draw_clustered_points_on_image(np.array(sample.image), lp_wg_ordered_by_angle)
    #angle_on_img.show()

    # draw stixel
    gt_stixel_img = draw_stixels_on_image(np.array(sample.image),grid_stixel)
    gt_stixel_img.show()

    # single scanline
    if config['explore_data']:
        column = Scanline(lp_wg_ordered_by_angle[96])
        obj_scanline_on_img = draw_obj_points_on_image(np.array(sample.image), column.objects, column.get_stixels())
        #obj_scanline_on_img.show()
        #draw_obj_points_2d(column.objects, column.get_stixels())
    print(len(grid_stixel))

if __name__ == "__main__":
    main()
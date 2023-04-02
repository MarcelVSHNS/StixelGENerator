import yaml

from dataloader.WaymoDataset import WaymoDataLoader
from libraries.visualization import plot_points_on_image


def main():
    with open('config.yaml') as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    data_dir = config['dataset']['raw_path']['train']
    print(f'path: {data_dir}')
    waymo_dataset_training = WaymoDataLoader(data_dir=data_dir, camera_segmentation_only=True)

    # Example to display the lidar camera projection
    sample = waymo_dataset_training[723]
    for i in range(5):
        plot_points_on_image(images=sample[0].images,
                             laser_points=sample[0].laser_points,
                             laser_camera_projections=sample[0].laser_camera_projections,
                             view=i)


if __name__ == "__main__":
    main()

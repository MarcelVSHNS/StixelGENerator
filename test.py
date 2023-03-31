from dataloader.WaymoDataset import WaymoDataLoader
from libraries.visualization import plot_points_on_image


def main():
    # sample tfrecord: tfrecord_file='individual_files_testing_segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord'
    data = WaymoDataLoader(data_dir='raw/training',
                           tfrecord_file=None,
                           camera_segmentation_only=True,
                           shuffle=True)
    for i in range(5):
        plot_points_on_image(data.frames[0], view=i)


if __name__ == "__main__":
    main()

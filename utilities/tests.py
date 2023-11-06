from dataloader.AmeiseDataset import AmeiseDataLoader
import ameisedataset as ad
import matplotlib.pyplot as plt


def main():
    # load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
    #     .images (5 views) and .laser_points (top lidar, divided into 5 fitting views).
    ameise_dataset = AmeiseDataLoader(data_dir='raw/ameise/validation', first_only=True)
    # Usage: sample = waymo_dataset[idx][frame_num]
    sample = ameise_dataset[5][-1]

    img_l = sample.cameras[ad.data.Camera.STEREO_LEFT].image
    img_r = sample.cameras[ad.data.Camera.STEREO_RIGHT].image

    disparity_map = ad.utils.create_disparity_map(img_l, img_r)
    plt.figure(figsize=(20, 12))
    plt.imshow(disparity_map, cmap='viridis')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
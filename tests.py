from dataloader.AmeiseDataset import AmeiseDataLoader



def main():
    # load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
    #     .images (5 views) and .laser_points (top lidar, divided into 5 fitting views).
    ameise_dataset = AmeiseDataLoader(data_dir='raw/ameise/', first_only=False)
    # Usage: sample = waymo_dataset[idx][frame_num]
    sample = ameise_dataset[-1][-1]
    print("hi!")

if __name__ == "__main__":
    main()
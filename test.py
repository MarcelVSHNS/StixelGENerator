from dataloader.WaymoDataset import WaymoDataLoader


def main():
    data = WaymoDataLoader(tf_record_files='individual_files_testing_segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord',
                           data_dir='raw/training')
    print(data.camera_segmentation_only)


if __name__ == "__main__":
    main()

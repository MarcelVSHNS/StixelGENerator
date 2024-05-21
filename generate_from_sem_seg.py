import os.path

from dataloader.CityscapesDataset import CityscapesDataLoader as Dataloader
from generate import _export_single_dataset
import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    if config['SemSeg']['for_dataset'] == "ameise":
        img_size = (1920, 1200)
    elif config['SemSeg']['for_dataset'] == "kitti":
        img_size = (1248, 376)
    else:
        raise Exception("Specify the dataset!")

    cs_dataset = Dataloader(root_dir=config['raw_data_path'],
                            img_size=img_size,
                            grid_step=config['grid_step'])
    dataset_path = os.path.join(cs_dataset.name, config['SemSeg']['for_dataset'])
    for sample in cs_dataset:
        _export_single_dataset(image_left=sample.left_img,
                               image_right=sample.right_img,
                               stixels=sample.gt_obstacles,
                               dataset_name=dataset_path,
                               name=sample.name,
                               export_phase="testing")
        print(f"{sample.name} exported.")
    print("Finished exporting dataset.")


if __name__ == "__main__":
    main()

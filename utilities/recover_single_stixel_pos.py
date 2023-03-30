import pandas as pd
import os
import yaml


with open('/home/marcel/workspace/groundtruth_stixel/config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
phase = config['dataset']['raw_path']['val']            # validation
output_phase = config['dataset']['raw_path']['test']    # testing
output_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'], output_phase)     # data/test
input_image_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'], phase)       # data/val
input_annotations_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'])        # data
data_set = config['dataset']['set_name']


def main():
    new_dataset_part = 40
    annotations = pd.read_csv(os.path.join(input_annotations_path, phase + "_data.csv"))
    img_map = create_image_reference_map(annotations)
    start_idx = int(len(img_map) - len(img_map)*40/100)
    for img in img_map:
        label = annotations.groupby('img_path').get_group(img)
        img_name = os.path.basename(img)
        label.to_csv(os.path.join(input_image_path, "single_stixel_pos", os.path.splitext(img_name)[0] + ".txt"),
                     header=False, index=False)
        print(img)
        # take every 10th image/annotation pair


def create_image_reference_map(annotation_file):
    # Separate the path col
    image_map = annotation_file.loc[:, 'img_path'].tolist()
    # drop all duplicates and add a label array
    image_map = list(dict.fromkeys(image_map))
    return image_map


if __name__ == "__main__":
    main()

from zipfile import ZipFile
import os
from pathlib import Path
from os.path import basename
import yaml


with open('/home/marcel/workspace/groundtruth_stixel/config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
phase = config['dataset']['raw_path']['test']
output_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'], phase + "_compressed")
input_image_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'], phase)
input_annotations_path = os.path.join(input_image_path, "single_stixel_pos")
data_set = config['dataset']['set_name']


def main():
    num_packages = 3
    single_stixel_pos_export = os.path.isdir(input_annotations_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    img_folder_list = os.listdir(input_image_path)
    image_list = [f for f in img_folder_list if f.endswith('.png')]
    n_sized = int(len(image_list) / num_packages)
    image_chunks = chunks(image_list, n_sized)

    n = 0
    for img_list in image_chunks:
        # create a ZipFile object
        annotZip = ZipFile(os.path.join(output_path,
                                        phase + '_' + data_set + '_compressed_single_annotations_' + str(n) + '.zip'), 'w')
        with ZipFile(os.path.join(output_path, phase + '_' + data_set + '_compressed_' + str(n) + '.zip'), 'w') as zipObj:
            for img in img_list:
                # create complete filepath of file in directory
                file_path = os.path.join(input_image_path, img)
                zipObj.write(file_path, basename(file_path))
                if single_stixel_pos_export:
                    annot = os.path.splitext(img)[0] + '.txt'
                    annot_file_path = os.path.join(input_annotations_path, annot)
                    annotZip.write(annot_file_path, basename(annot_file_path))
        n += 1
        annotZip.close()
        print("zip num: " + str(n) + " from total " + str(num_packages) + " created!")


def chunks(lst, n):
    # Yield successive n-sized chunks from lst.
    # e.g. n=950 with lst=20.000 means 20 chunks
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    main()

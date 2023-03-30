import os
import yaml


with open('/home/marcel/workspace/groundtruth_stixel/config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
phase = config['dataset']['raw_path']['test']
output_path = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'])
input_path = os.path.join(output_path, phase, "targets")


def main():
    image_list = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    header = "img_path,x,y,class\n"

    data = []
    for file_name in image_list:
        with open(os.path.join(input_path, file_name)) as fp:
            data.append(fp.read())

    # os.chdir(path)
    with open(os.path.join(output_path, phase + "_data.txt"), 'w') as fp:
        fp.write(''.join(data))


if __name__ == "__main__":
    main()

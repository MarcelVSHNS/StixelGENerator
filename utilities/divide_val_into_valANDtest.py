import os
import shutil


def main():
    val_path = "/home/marcel/workspace/groundtruth_stixel/raw/validation"
    test_path = "/home/marcel/workspace/groundtruth_stixel/raw/testing"
    label_folder = "single_stixel_pos"
    val_image_list = [f for f in os.listdir(os.path.join(os.getcwd(), val_path)) if f.endswith('.png')]
    num_parts = 3

    # divide into 3 equal parts
    test_data = list(chunks(val_image_list, int(len(val_image_list)/num_parts)))

    # take the middle one (1/3 of the val raw) for testing
    for file_name in test_data[1]:
        # move image
        shutil.move(os.path.join(val_path, file_name), os.path.join(test_path, file_name))
        print(file_name + " moved. \n")
        # move relative gt label
        shutil.move(os.path.join(val_path, label_folder, os.path.splitext(file_name)[0] + ".txt"),
                    os.path.join(test_path, label_folder, os.path.splitext(file_name)[0] + ".txt"))
        print(os.path.splitext(file_name)[0] + ".txt moved. \n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    main()

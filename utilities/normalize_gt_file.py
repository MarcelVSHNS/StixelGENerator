import os
import csv
import numpy as np


path_to_gt_file = "data/validation_data.csv"
output_name = "validation_data.csv"
file = os.path.join(os.getcwd(), path_to_gt_file)
header = ['img_path', 'x', 'y', 'class']


def normalize_into_grid(pos, step=8, offset=0):
    val_norm = 0
    rest = pos % step
    if rest > step / 2:
        val_norm = pos + (step - rest)
    else:
        val_norm = pos - rest
    assert val_norm % step == 0
    return val_norm + offset


def main():
    lines = [line.rstrip("\n") for line in open(file, "r")]
    assert len(lines) > 0
    lines = [line.split(",") for line in lines]
    # e.g.:  testing/segment-8956556778987472864_3404_790_3424_790_with_camera_labels-24-1.png 0 1251 2

    for pos_row in lines:
        #pos_row[2] = normalize_into_grid(int(pos_row[2]))
        #if pos_row[2] == 1280:
        #    pos_row[2] = 1272
        # pos_row[3] = int(pos_row[3]) + 1
        # /home/marcel/workspace/groundtruth_stixel/dataset_validation/segment-5372281728627437618_2005_000_2025_000_with_camera_labels-37-2.png
        pos_row[0] = pos_row[0].split("/")[-1]
        # pos_row[0] = "testing/" + path[-1]


    """
    test = np.asarray(lines)
    print("X_val MAX: " + str(np.amax(np.asarray(test[..., 1], dtype=int))))
    print("X_val MIN: " + str(np.amin(np.asarray(test[..., 1], dtype=int))))
    print("Y_val MAX: " + str(np.amax(np.asarray(test[..., 2], dtype=int))))
    print("Y_val MIN: " + str(np.amin(np.asarray(test[..., 2], dtype=int))))
    """

    with open(output_name, 'w') as f:
        write = csv.writer(f)
        # write.writerow(header)
        write.writerows(lines)

    print("Done!")


if __name__ == "__main__":
    main()

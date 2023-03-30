import os
import pandas as pd
import csv

new_header = ["img_path", "x", "y", "class"]

def main():
    if False:
        df = pd.read_csv('data/validation_data.csv')
        image_map = df.loc[:, 'img_path'].tolist()
        # drop all duplicates and add a label array
        image_map = list(dict.fromkeys(image_map))
        image_map = pd.DataFrame(image_map)
        image_map.to_csv("test.csv", index=False, header=False)

    if False:
        files = os.listdir("data/testing/single_stixel_pos")
        for file in files:
            # file = "data/testing/single_stixel_pos/segment-933621182106051783_4160_000_4180_000_with_camera_labels-2-0.txt"
            lines = [line.rstrip("\n") for line in open("data/testing/single_stixel_pos/" + file, "r")]
            assert len(lines) > 0
            lines = [line.split(",") for line in lines]
            # e.g.:  testing/segment-8956556778987472864_3404_790_3424_790_with_camera_labels-24-1.png 0 1251 2

            for pos_row in lines:
                # pos_row[2] = normalize_into_grid(int(pos_row[2]))
                # if pos_row[2] == 1280:
                #    pos_row[2] = 1272
                # pos_row[3] = int(pos_row[3]) + 1
                # /home/marcel/workspace/groundtruth_stixel/dataset_validation/segment-5372281728627437618_2005_000_2025_000_with_camera_labels-37-2.png
                pos_row[0] = "testing/" + pos_row[0].split("/")[1]

            with open("data/testing/targets/" + os.path.splitext(file)[0] + ".csv", 'w') as f:
                write = csv.writer(f)
                write.writerows(lines)

    if False:
        files = os.listdir("data/testing/targets")

        for file in files:
            path = os.path.join("data/testing/targets", file)
            df = pd.read_csv(path, header=None, names=["img_path", "x", "y", "class"])
            df.to_csv(os.path.join("data/testing/targets", os.path.splitext(file)[0] + ".csv"), index=False)
            print(f'{os.path.splitext(file)[0]} finished.')

    if True:
        files = os.listdir("data/testing/targets")
        map = []
        for file in files:
            map.append(os.path.splitext(file)[0])
        image_map = pd.DataFrame(map)
        image_map.to_csv("test.csv", index=False, header=False)


    print("Done!")


if __name__ == "__main__":
    main()

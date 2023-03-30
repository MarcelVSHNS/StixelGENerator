import os

def main():
    path = "validation"

    fp = open('validation_data.txt', 'r')
    gt_lines = fp.readlines()

    new_data = []
    for gt_line in gt_lines:
        new_data.append(os.path.join(path, os.path.basename(gt_line)))

    with open(path + "_data_new.txt", 'w') as fp:
        fp.write(''.join(new_data))


if __name__ == "__main__":
    main()

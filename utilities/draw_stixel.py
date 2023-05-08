import cv2
import os
import matplotlib.pyplot as plt
import yaml

"""
    TODO:
    - read a folder with e.g. training data, continue with >return<
    - read the fitting stixel from one file
"""

wantToSave = False
os.chdir('..')
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
stixelWidth = config['stixel']['width']
stixelHeight = config['stixel']['height']
path = "dataset_validation"
thickness = 1
idx_list = [1,2,3]
image_list = [f for f in os.listdir(path) if f.endswith('.png')]

for idx in idx_list:
    img = cv2.imread(os.path.join(path, image_list[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fig = plt.figure(figsize=(20, 12))

    color = (0, 255, 0)
    f = open(os.path.join(path, "single_stixel_pos", os.path.splitext(image_list[idx])[0] + ".txt"), "r")
    stixel = f.readlines()

    # gts means groundTruthStixel
    for gts in stixel:
        x = int(gts.split()[1])
        y = int(gts.split()[2])
        # starts top left corner
        start_point = (x, y-stixelHeight)
        # ends bottom right corner
        end_point = (x+stixelWidth, y)
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

    plt.imshow(img)
    plt.show()

    if wantToSave:
        cv2.imwrite("result/" + os.path.splitext(image_list[idx])[0] + "-result.png", img)


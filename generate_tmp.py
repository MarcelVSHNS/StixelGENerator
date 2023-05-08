import os
import cv2
from pathlib import Path
# import pptk
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from dataloader.WaymoDataset import WaymoFrame
import yaml
import threading

""" TODO:
 - Offer an automatic conversion of all data (split into train, val, test) + just a test folder
 - Add Threading to generate training data
 - Improve border detection with function estimation, more precise points (interpolation)
 - Improve documentation
 - Add usage description
"""


""" Generate LiDAR based Labels for StixelNet (Waymo Open Dataset)
    Input:
        - list of laser points in form: [N,(x,y,z)]
        - relative front view image
        - calibration data from laser to image
    Procedure:
        1: Extract laser data from FRAME                    (x) <- automate with Docker Container
        2: Combine all laser points and cut the front view  (x)
        3: Find object and floor borders                    (x) <- IMPROVE
        4: Project laser points on image                    (x)
        5: Extract object and floor borders image position  (x)
        6: Export point list to file and folder (with image)(x)
    Returns: 
        A text file in form: path/filename.png     x-coordinate    y-coordinate
            e.g. 00000011.png 40 282
"""
print(os.getcwd())
with open('/home/marcel/workspace/groundtruth_stixel/config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
phase = config['dataset']['raw_path']['train']
data_src = os.path.join(config['dataset']['raw_data_base_folder'], phase)
output_data_base_folder = os.path.join(os.getcwd(), config['dataset']['output_data_base_folder'])
output_data_folder = os.path.join(output_data_base_folder, phase)
set_name = config['dataset']['set_name']


def main():
    filename = config['investigation']['filename']
    explore_data = False
    num_views = 5
    num_threads = 10
    # List: 120,142 - 5, 143
    frame_num_from_set = 0
    # Cuts for just the front view (front = 0, front_left = 1, front_right = 2, side_left = 3, side_right = 4)
    view = 0
    # num of angles while 0 is left and 384 is right (assumed the width is 1920 px and the col_width is 5)
    sample_angle = 400

    if explore_data:
        # Load test data
        frames = unpack_tfrecord_file_from_path("training" + "/" + filename)

        # Check the given specific frame number
        if frame_num_from_set <= len(frames):
            investigated_frame = WaymoFrame(frames[frame_num_from_set], view)
        else:
            # take the first/ only available one
            investigated_frame = WaymoFrame(frames[0], view)

        investigated_frame.print_frame_info()
        investigated_frame.print_all_annotated_stixel_on_image()
        #investigated_frame.print_all_projected_points_on_image()
        #investigated_frame.print_object_cuts_per_col_on_img(col=col)
        investigated_frame.print_object_cuts_per_col_on_img()
        #investigated_frame.print_all_segmentation_points_on_image()
        #investigated_frame.print_all_segmented_pixel_on_image()
        #investigated_frame.print_estimated_object_borders_on_image()
        #investigated_frame.print_estimated_stixel_preview_on_image()
        #investigated_frame.print_single_angle_2d(angle)
        #investigated_frame.print_single_angle_3d_on_image(angle)

    if not explore_data:
        folder_list = os.listdir(data_src)
        dataset_list_full = [f for f in folder_list if f.endswith('.tfrecord')]
        print("Found " + str(len(dataset_list_full)) + " data set files")
        Path(os.path.join(os.getcwd(), output_data_folder, "single_stixel_pos")).mkdir(parents=True, exist_ok=True)
        thread_workload = int(len(dataset_list_full) / num_threads)
        assert thread_workload > 0, "Too less files for num_threads"
        dataset_chunk = list(chunks(dataset_list_full, thread_workload))
        threads = []
        for dataset_list in dataset_chunk:
            # create thread with arg dataset_list (chunk)
            x = threading.Thread(target=export_frame_data, args=(dataset_list,))
            threads.append(x)
            x.start()
            print("Thread is working ...")
            # export_frame_data(frame_list, dataset_name, num_views=num_views)
        for thread in threads:
            thread.join()
        # concatenate_single_stixel_poses_text_files()
        print("Finished!")
    return ()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def export_frame_data(dataset_list, num_views=3):
    for dataset_name in dataset_list:
        # unpack every used frame from set and store in list
        frame_list = unpack_tfrecord_file_from_path(data_src + "/" + dataset_name)
        frame_num = 0
        for frame in frame_list:
            for camera_view in range(num_views):
                if frame.images[camera_view].camera_segmentation_label.panoptic_label:
                    frame_data = WaymoFrame(frame, camera_view)
                    training_points = frame_data.create_training_data_from_img_lidar_pair(reverse_img_reference=True)
                    # Write Training and save image
                    img_name = os.path.splitext(dataset_name)[0] + "-" + str(
                        frame_num) + "-" + str(camera_view) + ".png"
                    point_strings = []
                    # Create training File in format: path/filename.png     x-coordinate    y-coordinate
                    for point in training_points:
                        point_strings.append(
                            phase + "/" + img_name + "," + str(point[0]) + "," + str(int(point[1])) + "," + str(int(point[2])) + "\n")
                    with open(os.path.join(output_data_folder, "targets",
                                           os.path.splitext(img_name)[0] + ".csv"), 'w') as fp:
                        fp.write("img_path,x,y,class\n")
                        fp.write(''.join(point_strings))
                    decoded_image = tf.io.decode_jpeg(frame_data.images[camera_view].image, channels=3,
                                                      dct_method='INTEGER_ACCURATE')
                    decoded_image = cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_data_folder + "/" + img_name, decoded_image)
                    print(img_name + " finished")
            frame_num += 1
    print("Chunk finished!")


def concatenate_single_stixel_poses_text_files():
    """
    Merges all single GT data per image into one file, which fits for the StixelNet NN.
    It is assumed that inside the training_data_folder the "single_stixel_pos"-folder exists
    Args:
        training_data_folder: forwarding the source of the txt files #
        set_name: naming
    """
    header = "img_path,x,y,class\n"
    txt_file_subfolder = "single_stixel_pos"
    path = os.path.join(output_data_folder, txt_file_subfolder)
    image_list = [f for f in os.listdir(path) if f.endswith('.txt')]

    data = []
    data.append(header)
    for file_name in image_list:
        with open(os.path.join(path, file_name)) as fp:
            data.append(fp.read())

    with open(os.path.join(output_data_base_folder, phase + "_data.csv"), 'w') as fp:
        fp.write(''.join(data))


if __name__ == "__main__":
    main()

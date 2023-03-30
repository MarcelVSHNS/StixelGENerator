import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libraries import PerceptualObject

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils








def convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=0):
    """Convert segmentation labels from range images to point clouds.
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.
    Returns:
      point_labels: {[N, 2]} list of 3d lidar points segmentation labels. 0 for
        points that are not labeled."""
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


# TODO: use image attributes instead of static parameter
def get_spherical_point_cloud_with_fixed_angles(points, img_width=1920, col_width=8):
    """
    Args:
        points:
        img_width:
        col_width:
    Returns: (N, [r, az, el])
    """
    spherical = np.copy(funcLib3D.transform_cartesian_to_spherical(points))
    num_col = img_width/col_width*2+0.0304857694325
    grid_step = 0.84 / num_col
    for pt in spherical:
        pt[1] = pt[1] - (pt[1] % grid_step)
    return spherical


class WaymoDataLoader(object):
    def __init__(self, tf_record):
        self.images = sorted(tf_frame.images, key=lambda i: i.name)

    def unpack_tfrecord_file_from_path(self, tf_record_filename, segmentation_only=True):
        """
        Loads a tf-record file from the given path and returns a list of frames from the file
        Args:
            tf_record_filename: a given relative or absolute path to the file
            segmentation_only: if True, only frames with segmentation data will be used
        Returns: an array with FRAMES from the tfrecord file, in case of a not tf file it returns an empty array
        """
        if tf_record_filename.endswith('.tfrecord'):
            dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
            frame_list = []
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if segmentation_only:
                    # if frame.lasers[0].ri_return1.segmentation_label_compressed:
                    #    frame_list.append(frame)
                    if frame.images[0].camera_segmentation_label.panoptic_label:
                        frame_list.append(frame)
                else:
                    frame_list.append(frame)
            print("Num_seg_frames: " + str(len(frame_list)))
            return frame_list
        else:
            return []

class WaymoData(object):
    def __init__(self, tf_frame, view=None):
        self.images = sorted(tf_frame.images, key=lambda i: i.name)
        self.frame = tf_frame
        self.view = view
        if self.frame.images[view].camera_segmentation_label.panoptic_label:
            self.camera_labels, self.camera_instance_labels = self.cut_panoramic_segmentation_labels()
        else:
            self.camera_labels = np.array(False)
            self.camera_instance_labels = np.array(False)
        self.points, self.projections, self.laser_labels = cut_point_cloud_to_view(tf_frame, self.view)
        # TODO: Try extracting polar coordinates instead of a transformation of cartesian
        self.points_spherical = get_spherical_point_cloud_with_fixed_angles(self.points)
        self.angle_list = self.__get_list_of_angles()
        # TODO: add the range_image_top_pose here
        self.laser_height = 1.8

    def print_frame_info(self):
        print("Angles: " + str(len(self.angle_list)))
        # print(self.segLabels.shape)

    def cut_panoramic_segmentation_labels(self):
        # here 2 and 3 were switched
        # self.views: (front = 0, front_left = 1, side_left = 2!!, front_right = 3!!, side_right = 4)
        if self.view == 2 or self.view == 3:
            if self.view == 2:
                self.view = 3
            else:
                self.view = 2
        # images = sorted(tf_frame.images, key=lambda i: i.name)
        segmentation_proto = [image.camera_segmentation_label for image in self.frame.images]

        panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
            segmentation_proto[self.view]
        )
        semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic_label,
            segmentation_proto[self.view].panoptic_label_divisor
        )
        return semantic_label, instance_label

    def cut_point_cloud_to_view(self, view=0):
        # Cuts for just the front view (front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4)
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(self.frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            self.frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # 3d points in vehicle frame.
        points_all = points[0]
        # points_all = np.concatenate(points, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = cp_points[0]
        # cp_points_all = np.concatenate(cp_points, axis=0)

        # Sort images by name
        images = sorted(tf_frame.images, key=lambda i: i.name)

        # Convert into a tf_v2 tensor (projections)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        # define mask where the projections equal the picture view
        # (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        # cp_points_all_tensor[..., 0] while 0 means cameraName.name (first projection)
        mask = tf.equal(cp_points_all_tensor[..., 0], images[view].name)

        # transforms the projected points after slicing it from the mask into float values
        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all = tf.gather_nd(points_all, tf.where(mask))

        """    NOTE:
        1. Only TOP LiDAR has segmentation labels.
        2. Not every frame has segmentation labels. This field is not set if a
           frame is not labeled.
        3. There can be points missing segmentation labels within a labeled frame.
            Their label are set to TYPE_NOT_LABELED when that happens."""
        if segmentation_labels:
            point_labels = convert_range_image_to_point_cloud_labels(tf_frame, range_images, segmentation_labels)
            point_labels_all = point_labels[0]
            point_labels_all = tf.gather_nd(point_labels_all, tf.where(mask))
            return points_all.numpy(), cp_points_all_tensor.numpy(), point_labels_all.numpy()

        return points_all.numpy(), cp_points_all_tensor.numpy(), np.array(False)

    def print_single_angle_2d(self, angle, marker_size=1):
        """
        Prints a 2D Graph of an angle and indicates the found borders
        :param angle:
        :param marker_size:
        :return:
        """
        xs = []
        zs = []
        gs = []
        angle_indices = self.__get_index_list_by_angle(self.angle_list[angle])
        for index in angle_indices:
            xs.append(self.points[index][0])
            zs.append(self.points[index][2])
        objects, detail_list = self.__extract_borders_from_one_column(self.angle_list[angle], research_active=True)
        for point in detail_list:
            gs.append(point[4])
        # 'bo' draws point without a connecting line
        plt.figure(figsize=(15, 10))
        plt.plot(xs, zs, markersize=0.5)
        plt.plot(xs, zs, 'bo', markersize=marker_size)
        if objects.size != 0:
            plt.plot(objects[..., [0]], objects[..., [2]], 'ro', markersize=marker_size + 2)
        for x, y, g in zip(xs, zs, gs):
            label = "{:.3f}".format(g)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(12, 10),  # distance from text to points (x,y)
                         size=7,
                         ha='center')
        # change axis dims
        if objects.size != 0:
            plt.axis([objects[0][0]*0.5, objects[0][0]*2, -1, 2.5])
        else:
            plt.axis([4, 50, -0.4, 2.5])
        plt.title(np.degrees(angle))
        plt.show()
        return None

    def print_single_angle_3d_on_image(self, angle):
        angle_indices = self.__get_index_list_by_angle(self.angle_list[angle])
        # just use points with index ...
        angle_points = self.__cut_points_by_index(self.points, angle_indices)
        cp_angle_points = self.__cut_points_by_index(self.projections, angle_indices)
        # combine projections and measurement
        projected_angle_point_line = self.__apply_image_projections_on_point_cloud(angle_points, cp_angle_points)
        # Image with one angle height profile (laser)
        borders = []
        objects = self.__extract_borders_from_one_column(self.angle_list[angle])
        for obj in objects:
            borders.append(self.__apply_image_projections_on_point_cloud(self.points[int(obj[3])], self.projections[int(obj[3])]))
        funcLibImg.plot_points_on_image(projected_angle_point_line, self.images[self.view], funcLibImg.rgba, objects=borders)

    def print_all_projected_points_on_image(self):
        merged_points = self.__apply_image_projections_on_point_cloud(self.points, self.projections)
        funcLibImg.plot_points_on_image(merged_points, self.images[self.view], funcLibImg.rgba)

    def print_all_segmentation_points_on_image(self):
        if self.laser_labels.any():
            merged_points = np.column_stack([self.projections[..., 1:3], self.laser_labels[..., 1]])
            funcLibImg.plot_points_on_image(merged_points, self.images[self.view], funcLibImg.rgb_seg)
        else:
            print("No Laser-segmentation Data available")

    def print_all_segmented_pixel_on_image(self):
        if self.camera_labels.any():
            funcLibImg.plot_segmented_image(self.camera_labels, self.camera_instance_labels)
        else:
            print("No Panoptic-segmentation Data available")

    def print_estimated_object_borders_on_image(self):
        object_borders = self.__get_object_borders_point_cloud_from_frame(bottom_select=True)
        funcLibImg.plot_points_on_image(object_borders, self.images[self.view], funcLibImg.rgba)

    def print_estimated_stixel_preview_on_image(self):
        object_borders = self.__get_object_borders_point_cloud_from_frame(bottom_select=True)
        stixel_points = self.__select_bottom_point_per_col(object_borders)
        funcLibImg.plot_points_on_image(stixel_points, self.images[self.view], funcLibImg.rgba, with_range=False)
        # funcLibImg.plot_stixel_on_image(stixel_points, self.images[0])

    def print_object_cuts_per_col_on_img(self, img_width=1920, col=-1):
        all_cuts = []
        if col == -1:
            for i in range(4, img_width-1, 8):
                cuts_per_col = self.__find_object_cuts_per_col_based_on_camera_segmentation(col=i, img_height=self.camera_labels.shape[0])
                all_cuts = all_cuts + cuts_per_col
        else:
            all_cuts = self.__find_object_cuts_per_col_based_on_camera_segmentation(col=col, img_height=self.camera_labels.shape[0])
            stixel_from_col = self.__extract_stixel_per_col(col, explore=True, img_height=1280)
            funcLibImg.plot_stixel_on_image(stixel_from_col, self.images[self.view])
        funcLibImg.plot_list_of_cuts_on_image(self.images[self.view], all_cuts)

    def print_all_annotated_stixel_on_image(self, img_width=1920):
        #img_height = tf.image.decode_jpeg(self.images[self.view].image).shape[0]
        all_stixels = []
        for i in range(4, img_width - 1, 8):
            stixel_per_col = self.__extract_stixel_per_col(col=i, img_height=self.camera_labels.shape[0])
            if stixel_per_col is not None:
                all_stixels = all_stixels + stixel_per_col
        funcLibImg.plot_stixel_on_image(all_stixels, self.images[self.view])

    def create_training_data_from_img_lidar_pair(self, bins=160, img_width=1920, reverse_img_reference=False):
        """
        object_borders = self.__get_object_borders_point_cloud_from_frame(bottom_select=True)
        if reverse_img_reference:
            bottom_points = self.__select_bottom_point_per_col(object_borders)

            step = height / bins
            training_points = np.asarray(bottom_points)
            # adapt horizontal num of possible cuts -
            # Quantization of the estimated object borders to adapt the num of bins for the NN
            for pt in training_points:
                rest = pt[1] % step
                if rest >= step / 2:
                    pt[1] = pt[1] + (step - rest)
                else:
                    pt[1] = pt[1] - rest
                assert pt[1] % step == 0
            return funcLibImg.change_reference_point_on_image(training_points)
        else:
            return self.__select_bottom_point_per_col(object_borders)
        """
        all_cuts = []
        for i in range(0, self.camera_labels.shape[1] - 1, 8):
            cuts_per_col = self.__find_object_cuts_per_col_based_on_camera_segmentation(col=i,
                                                                                        img_height=self.camera_labels.shape[0])
            all_cuts = all_cuts + cuts_per_col
        return all_cuts


    def __extract_borders_from_one_column(self, angle, research_active=False):
        """ This function takes a list of points to iterate over the angle/ column to find the highest steep between
            relative points to spot out a possible object border (the lower one).
                      Args:
                        angle: the angle from the angle_indices list in number format >just for information<
                        research_active: if True print probability list with gradient values, if False nothing
                      Returns:
                        An array with calculated (estimated) borders of objects
        """
        angle_indices = self.__get_index_list_by_angle(angle)
        pts_angle_list = []
        for index in angle_indices:
            pts_angle_list.append(self.points[index])
        pts_angle_list = np.column_stack((pts_angle_list, angle_indices))
        # calculate gradient
        gradient_list = []
        for i in range(len(pts_angle_list) - 1):
            gradient_list.append(np.absolute(funcLib3D.calc_gradient(pts_angle_list[i], pts_angle_list[i + 1])))
        gradient_list.append(0)
        pts_angle_list = np.column_stack((pts_angle_list, gradient_list))
        # calculate border probability
        prob = []
        for value in gradient_list:
            prob.append(funcLib3D.calc_probability(value))
        pts_angle_list = np.column_stack((pts_angle_list, prob))
        if research_active:
            for i in range(len(gradient_list)):
                print("{idx}__  \tg:{grad} \t\tP:{prob}".format(idx=i, grad=round(gradient_list[i], 5),
                                                                prob=round(prob[i], 5)))
        # extract high probabilities
        high_prob_points = []
        for i in range(len(pts_angle_list)):
            # @param
            if pts_angle_list[i][5] >= 0.96:
                high_prob_points.append(pts_angle_list[i])

        # cluster objects based on x-difference
        if high_prob_points:
            object_cluster_threshold = 4.0
            object_list = []
            high_prob_points = np.array(high_prob_points)
            object_list.append(high_prob_points[0])
            for pt in high_prob_points:
                if pt[0] - object_cluster_threshold >= object_list[-1][0]:
                    object_list.append(pt)
            objects = np.array(object_list)
            if research_active:
                return objects, pts_angle_list
            else:
                return objects
        else:
            if research_active:
                print("no border found at: " + str(angle))
                return np.array([]), np.array([])
            return np.array([])

    def __get_index_list_by_angle(self, angle):
        """From all spherical points calculate every fitting angle (azimuth) point, sorted by elevation.
                  Args:
                    angle: azimuth angle in rad
                  Returns:
                    A list of indices with the given angle, related to the given point cloud, sorted by the height
                    (elevation)
        """
        # TODO: offset the cartesian coordinates by the sensor height before sorting by elevation
        list_angle = []
        for i in range(len(self.points_spherical)):
            # List of indices
            if self.points_spherical[i][1] == angle:
                list_angle.append(i)
        # use list in cartesian to shift all points by x
        pts_cartesian = []
        points_copy = np.copy(self.points)
        for index in list_angle:
            pts_cartesian.append(points_copy[index])
        offset = self.laser_height
        for pt in pts_cartesian:
            pt[2] = pt[2] - offset
        # transform new points into spherical
        points_sph_offset = funcLib3D.transform_cartesian_to_spherical(np.asarray(pts_cartesian))
        # order them by elevation
        index_list = []
        for i in range(len(list_angle)):
            index_list.append((list_angle[i], points_sph_offset[i][2]))
        index_list = sorted(index_list, key=lambda x: x[1])
        return [row[0] for row in index_list]

    def __get_object_borders_point_cloud_from_frame(self, bottom_select=False):
        # Find lower border points indices based on spherical coordinates
        # -values are right and positive are left
        object_border_list = []
        # for every angle
        for angle in self.angle_list:
            # extract the possible object borders
            objects = self.__extract_borders_from_one_column(angle)
            # if at least one is found
            if objects.size != 0:
                # for every found object...
                if bottom_select:
                    object_border_list.append(objects[0][3])
                else:
                    for obj in objects:
                        # store the index of it
                        object_border_list.append(obj[3])

        # Use the indices list with the border points to cut the point cloud and the relative projections
        object_points_cut = self.__cut_points_by_index(self.points, object_border_list)
        projection_points_cut = self.__cut_points_by_index(self.projections, object_border_list)
        # Use the cut list of points to combine them with the image
        projected_points_all = self.__apply_image_projections_on_point_cloud(object_points_cut, projection_points_cut)
        return projected_points_all

    def __get_list_of_angles(self):
        angles = []
        for pt in self.points_spherical:
            if pt[1] not in angles:
                angles.append(pt[1])
        return sorted(angles, reverse=True)

    def __apply_image_projections_on_point_cloud(self, points_view, cp_points_view):
        """
        Combines the distance and the relative position on an image
        :param points_view:
        :param cp_points_view:
        :return: a list of combined points: projections and depth in shape [..., (x, y, depth)]
        """
        # all lidar points normalized (length = 1, direction = same)
        # if normalized, the picture distance can be multiplied - otherwise the View-3D Cloud wil be kept
        points_view_norm = np.linalg.norm(points_view, axis=-1, keepdims=True)

        # combines all data from projection and the points into one array in numpy format
        # therefore take the width and height channel of the projection ([0] = name, [1] = x along width,
        # [2] = y along height)
        return np.concatenate([cp_points_view[..., 1:3], points_view_norm], axis=-1)

    def __cut_points_by_index(self, pts, index_list):
        """
        Applies a mask (index list) on a point list
        :param pts:
        :param index_list:
        :return:
        """
        pts_cut = []
        for index in index_list:
            pts_cut.append(pts[int(index)])
        return np.array(pts_cut)

    def __select_bottom_point_per_col(self, object_points, img_width=1920, col_width=8, margin=50):
        """
        Filter the extracted object borders by the bottom point for the StixelNet Training
        :param object_points: on a relative image with reference point top left
        :param img_width: image width
        :param col_width: Stixel width
        :param margin: Smoothing border which points from the bottom get calculated into the mean val in px
        :return: A list of points in format [N, 2] with inner dims [column height], reference is bottom left
        """
        # Convert points into the correct reference
        # projected_points = change_reference_point_on_image(projected_points)
        # for every col in the img
        # TODO: get from every col the max val and then use the mean to map the stixel width
        num_cols = img_width / col_width
        training_points = []
        for col in range(int(num_cols)):
            col_points = []
            for pt in object_points:
                if col * col_width <= pt[0] < col * col_width + col_width:
                    col_points.append(pt[1])
            if len(col_points) != 0:
                col_points_np = np.copy(np.asarray(col_points))
                for i in range(len(col_points)):
                    if col_points[i] < (np.amax(col_points) - margin):
                        col_points_np = np.delete(col_points, i)
                training_points.append((col * col_width, int(np.ceil(np.mean(col_points_np)))))
        return training_points

    def __extract_stixel_per_col(self, col=4, img_height=1280, explore=False):
        stixel_list = []
        col_cuts = self.__find_object_cuts_per_col_based_on_camera_segmentation(col, img_height=img_height)
        segmented_objects = self.__list_of_objects_by_col_cuts(col_cuts)
        if segmented_objects:
            laser_pts_of_objects = self.__get_laser_pts_of_an_object(segmented_objects)
            perceptual_object_list = []
            for i in range(len(segmented_objects)):
                perceptual_object_list.append(
                    PerceptualObject.PerceptualObject(segmented_objects[i][0], segmented_objects[i][1], segmented_objects[i][2], laser_pts_of_objects[i]))
            # Remove objects of width 0
            for obj in perceptual_object_list:
                if obj.ending_row == obj.starting_row:
                    perceptual_object_list.remove(obj)
            for obj in perceptual_object_list:
                obj.apply_anomaly_detection_and_linear_completion()
                obj.get_stixel_from_depth_profile()
                if explore:
                    # funcLibImg.plot_points_on_image([item for sublist in laser_pts_of_objects for item in sublist],
                    #                                self.images[self.view], funcLibImg.rgba)
                    obj.print_object_profile()
                for stxl in obj.stixels:
                    stixel_list.append(stxl)
            return stixel_list
        else:
            return None

    def __find_object_cuts_per_col_based_on_camera_segmentation(self, col=4, img_height=1280):
        """
        col:  define a specific column which will be analysed
        Returns: a list with cuts in the shape: [..., (col, row , cut-class)]
        Possible Cut-Classes are - 0:Bottom, 1:Swib, 2:Top
        """
        cuts = []
        for i in range(img_height-1, 1, -1):
            first_pixel = self.__check_label_of_interest(self.camera_labels[i][col])
            following_pixel = self.__check_label_of_interest(self.camera_labels[i-1][col])
            # Ground pixel are 0, Objects pixel are 1
            if first_pixel < following_pixel:
                # gt_class = 0                            # cut-class
                gt_class = self.camera_labels[i-1][col]   # segmentation-class
                cuts.append((col, self._fit_point_to_grid(i), gt_class))
            # if the object meets another object of interest
            if first_pixel == following_pixel and first_pixel == 1:
                # if the pixel has the same class ... like car to car
                if self.camera_labels[i][col] == self.camera_labels[i-1][col]:
                    # ...check if the instance differs or if it is the same object: like red car to blue car
                    if self.camera_instance_labels[i][col] != self.camera_instance_labels[i-1][col]:
                        # gt_class = 1                            # cut-class
                        gt_class = self.camera_labels[i-1][col]  # segmentation-class
                        cuts.append((col, self._fit_point_to_grid(i), gt_class))
                # if the labels are both objects, and it differs like e.g. car to tree
                else:
                    # gt_class = 1                            # cut-class
                    gt_class = self.camera_labels[i][col]  # segmentation-class
                    cuts.append((col, self._fit_point_to_grid(i), gt_class))
            # vice versa: from object to the sky e.g.
            if first_pixel > following_pixel:
                # gt_class = 2                            # cut-class
                gt_class = self.camera_labels[i][col]  # segmentation-class
                cuts.append((col, self._fit_point_to_grid(i), gt_class))
        return cuts

    def _fit_point_to_grid(self, pos, step=8, offset=0):
        val_norm = 0
        rest = pos % step
        if rest > step / 2:
            val_norm = pos + (step - rest)
        else:
            val_norm = pos - rest
        assert val_norm % step == 0
        if val_norm == 1280:
            val_norm = 1272
        return val_norm + offset

    def __check_label_of_interest(self, label):
        """
        Args:
            label: provide a segmentation label (an INT)
        Returns: "1" if it is an object of interest and "0" if ground object
        """
        if 19 < label < 23:
            return 0
        if 24 < label < 27:
            return 0
        return 1


    def __list_of_objects_by_col_cuts(self, col_cuts):
        """
        Analysis the col cuts (bottom, swib and Top) to extract objects and return a list with start- and endpoint
        Args:
            col_cuts: a list of cuts with the related cut-class (0:bottom, 1:swib, 2:top) [..., (col, row , cut-class)]
        Returns: A list of objects with a tuple of start and end point [..., (col, start, end)]
        """
        object_list = []
        for i in range(len(col_cuts)-1):
            # act only if cut-class is bottom or swib-bottom
            if col_cuts[i][2] == 1 and col_cuts[i+1][2] == 0:
                print("Swib to Bottom: check Col " + str(col_cuts[i][0]))
                return object_list

            # if bottom cut or swib-bottom
            if col_cuts[i][2] == 0 or col_cuts[i][2] == 1:
                # append (col,row - bottom) and (col, row - top)
                object_list.append((col_cuts[i][0], col_cuts[i][1], col_cuts[i+1][1]))
        print(object_list)
        return object_list

    def __get_laser_pts_of_an_object(self, objects, margin=3, grid=8):
        """
        Extract all pts from an object range within the related col.
        Merges all available measurements into one 8x8 grid.
        Args:
            objects: A list of objects with start and endpoint [..., (col, start, end)]
            margin: defines the +- area to search for related points
            grid: defines the steps between to measurement
        Returns: A list with lists of points per object [..., ([..., (x, y, d)])]
        """
        col_pts = []
        objects_pts = []
        col = objects[0][0]
        # all_pints is a list of combined points: projections and depth in shape [..., (col(x), row(y), depth)]
        all_points = self.__apply_image_projections_on_point_cloud(self.points, self.projections)

        # select just the points within the range of the col
        for pt in all_points:
            if col-margin < int(pt[0]) < col+margin:
                col_pts.append(list(pt))

        # iterate over all objects and create a list of object related points per entry
        for obj in objects:
            obj_pts = []
            for pt in col_pts:
                # if the row (x) of the objects start and end
                if obj[1] >= pt[1] >= obj[2]:
                    obj_pts.append(pt)
            objects_pts.append(obj_pts)

        # force the col sorted and object related pts into a 8x8 grid
        for obj in objects_pts:
            for pt in obj:
                rest = pt[1] % grid
                if rest >= grid / 2:
                    pt[1] = pt[1] + (grid - rest)
                else:
                    pt[1] = pt[1] - rest
                pt[0] = col
                assert pt[1] % grid == 0
        return objects_pts

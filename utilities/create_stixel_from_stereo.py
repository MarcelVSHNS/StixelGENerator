import numpy as np
import yaml
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.data.names import Camera, Lidar


with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
data_dir = config['raw_data_path'] + 'testing'

infos, frames = ameise.unpack_record("samples/frame.4mse")
pts, proj = tf.get_projection_matrix(frames[-1].lidar[Lidar.OS1_TOP].points, infos.lidar[Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]

im_rect_l = tf.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = tf.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])
disparity_map = tf.create_disparity_map(im_rect_l, im_rect_r)
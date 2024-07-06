import os
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict
from cityscapesscripts.helpers.labels import labels
from libraries import Stixel as BaseStixel


# id = name2label[name].id
name2label = {label.name: label for label in labels}
# name = id2label[id].name
id2label = {label.id: label for label in labels}
label_of_interest = ['building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
                     'traffic light', 'vegetation', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
                     'motorcycle', 'bicycle', 'train']

def crop_ego_vehicle(image, lower_pos):
    width, height = image.size
    return image.crop((0, 0, width, lower_pos))


class CityscapesData:
    def __init__(self, name, left_img: Image, right_img: Image, gt_labels: np.array, gt_colors: np.array, img_size=(1920, 1200), grid_step: int = 8):
        self.name = name
        self.gt_labels: np.array = gt_labels
        crop_lower = self.find_upper_border_of_ego_vehicle()
        self.original_left_img: Image = crop_ego_vehicle(left_img, crop_lower)
        self.original_right_img: Image = crop_ego_vehicle(right_img, crop_lower)
        self.img_size: Tuple[int, int] = img_size
        self.orig_img_size: Tuple[int, int] = self.original_left_img.size
        self.left_img: Image = self._crop_image_and_resize_to_size(self.original_left_img)
        self.right_img: Image = self._crop_image_and_resize_to_size(self.original_right_img)
        self.gt_colors: np.array = gt_colors
        self.grid_step: int = grid_step
        self.gt_obstacles: List[BaseStixel] = self._extract_obstacle_stixels()

    def find_upper_border_of_ego_vehicle(self):
        label = 1
        for row_index, row in enumerate(self.gt_labels):
            if label in row:
                return row_index
        return -1

    def _extract_obstacle_stixels(self) -> List[BaseStixel]:
        stixels: List[BaseStixel] = []
        edges = self._exclude_label(self.gt_labels)
        edges = self._find_edges(edges)
        edges = self._force_edges_to_grid(edges)
        edges = self._find_lower_border(edges)
        edges = self._crop_and_resize_label_array(edges)
        #img = self.draw_edges_on_image(self.left_img, edges)
        #img.show()
        ys, xs = np.where(edges == 1)
        for x, y in zip(xs, ys):
            stixels.append(BaseStixel(column=x // self.grid_step,
                                      top_row=y // self.grid_step,
                                      bottom_row=y // self.grid_step))
        return stixels

    def _crop_and_resize_label_array(self, label_array):
        scale_factor: float = label_array.shape[1] / label_array.shape[0]
        if self.img_size[0] / self.img_size[1] > scale_factor:
            scale_factor: float = label_array.shape[1] / self.img_size[0]
        else:
            scale_factor: float = label_array.shape[0] / self.img_size[1]
        width = label_array.shape[1]
        height = label_array.shape[0]
        new_width: int = int(self.img_size[0] * scale_factor)
        new_height: int = int(self.img_size[1] * scale_factor)
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        label_arr_cropped = label_array[y_offset:y_offset + new_height, x_offset:x_offset + new_width]
        # recalculate coordinates
        resized_label_array = np.zeros((self.img_size[1], self.img_size[0]))
        for x in range(label_arr_cropped.shape[1]):
            for y in range(label_arr_cropped.shape[0]):
                if label_arr_cropped[y, x] == 1:
                    new_x = int(x / scale_factor)
                    new_y = int(y / scale_factor)
                    resized_label_array[new_y, new_x] = 1
        return resized_label_array

    def _crop_image_and_resize_to_size(self, image: Image) -> Image:
        # resize to the fitting border by ratio
        scale_factor: float = image.width / image.height
        if self.img_size[0] / self.img_size[1] > scale_factor:
            image = image.resize((int(self.img_size[0]), int(self.img_size[0] / scale_factor)), Image.Resampling.LANCZOS)
        else:
            image = image.resize((int(self.img_size[1] * scale_factor), int(self.img_size[1])), Image.Resampling.LANCZOS)
        # crop image
        left: int = (image.size[0] - self.img_size[0]) // 2
        upper: int = (image.size[1] - self.img_size[1]) // 2
        right: int = left + self.img_size[0]
        lower: int = upper + self.img_size[1]
        cropped_image: Image = image.crop((left, upper, right, lower))
        return cropped_image

    def _force_edges_to_grid(self, edges):
        edges_grid = np.zeros(edges.shape)
        y, x = np.where(edges == 1)
        for y, x in zip(y, x):
            y_norm = self._normalize_into_grid(y, grid_step=self.grid_step)
            x_norm = self._normalize_into_grid(x, grid_step=self.grid_step)
            edges_grid[y_norm, x_norm] = 1
        return edges_grid

    def _find_lower_border(self, label_array):
        height = self.orig_img_size[1]  # 1024
        width = self.orig_img_size[0]   # 2048
        lower_edges = -1 * np.ones(width, dtype=int)
        for x in range(width):
            for y in range(height - 1, -1, -1):
                if label_array[y, x] == 1:
                    lower_edges[x] = y
                    break
        edges_array_lower_border = np.zeros((height, width))
        for x, y in enumerate(lower_edges):
            if y != -1:
                edges_array_lower_border[y, x] = 1
        return edges_array_lower_border

    @staticmethod
    def draw_edges_on_image(image, edges, color=(255, 0, 0)):
        if isinstance(image, Image.Image):
            image = np.array(image)
        edges_pos = np.where(edges == 1)
        image[edges_pos] = color
        return Image.fromarray(image)

    @staticmethod
    def _find_edges(label_array):
        edges = np.zeros_like(label_array)
        for y in range(1, label_array.shape[0] - 1):
            for x in range(1, label_array.shape[1] - 1):
                current_label = label_array[y, x]
                if (label_array[y + 1, x] != current_label or
                        label_array[y - 1, x] != current_label or
                        label_array[y, x + 1] != current_label or
                        label_array[y, x - 1] != current_label):
                    edges[y, x] = 1
        return edges

    @staticmethod
    def _exclude_label(label_array):
        label_of_no_interest = [label.id for label in labels if label.name not in label_of_interest]
        for label_id in label_of_no_interest:
            label_array[label_array == label_id] = 0
        return label_array

    @staticmethod
    def _include_label(label_array):
        for label_name in label_of_interest:
            label_array[label_array == name2label[label_name].id] = 1
        return label_array

    @staticmethod
    def _normalize_into_grid(pos: int, grid_step: int = 4):
        val_norm = pos - (pos % grid_step)
        return val_norm


class CityscapesDataLoader:
    def __init__(self, root_dir: str, img_size: Tuple[int, int] = (1920, 1200), grid_step=8):
        self.name: str = "cityscapes"
        self.root_dir: str = os.path.join(root_dir, self.name)
        self.left_img_path: str = os.path.join(self.root_dir, "leftImg8bit", "val")
        self.right_img_path: str = os.path.join(self.root_dir, "rightImg8bit", "val")
        self.gt_path: str = os.path.join(self.root_dir, "gtFine", "val")
        self.img_size: Tuple[int, int] = img_size
        self.grid_step : int = grid_step
        self.sample_map: List[Dict[str, str]] = self._read_cityscapes_data_structure()
        if len(self.sample_map) == 0:
            raise Exception("No Files found, check root folder!")

    def __len__(self) -> int:
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> CityscapesData:
        city: str = self.sample_map[idx]["city"]
        frame: str = self.sample_map[idx]["frame"]
        left_img: Image = Image.open(os.path.join(self.left_img_path, city, frame + "leftImg8bit" + ".png"))
        right_img: Image = Image.open(os.path.join(self.right_img_path, city, frame + "rightImg8bit" + ".png"))
        gt_labels: np.array = np.array(Image.open(os.path.join(self.gt_path, city, frame + "gtFine" + "_labelIds" + ".png")))
        gt_colors: np.array = np.array(Image.open(os.path.join(self.gt_path, city, frame + "gtFine" + "_color" + ".png")))
        cs_data: CityscapesData = CityscapesData(name=frame[:-1],
                                                 left_img=left_img,
                                                 right_img=right_img,
                                                 gt_labels=gt_labels,
                                                 gt_colors=gt_colors,
                                                 img_size=self.img_size,
                                                 grid_step=self.grid_step)
        return cs_data

    def __repr__(self):
        return f"[city]_[seq]_[frame]_[type]"

    def _read_cityscapes_data_structure(self, split='val', data_type='leftImg8bit', file_ending='.png') -> List[Dict[str, str]]:
        city_frame_list = []
        crop_length = len(data_type) + len(file_ending)
        sample_folder = os.path.join(self.root_dir, data_type, split)
        city_list = [name for name in os.listdir(sample_folder) if
                     os.path.isdir(os.path.join(sample_folder, name))]
        for city in city_list:
            frame_list = [name for name in os.listdir(os.path.join(sample_folder, city)) if name.endswith(file_ending)]
            for frame in frame_list:
                # berlin_000001_000019_leftImg8bit.png
                city_frame_list.append({"city": city, "frame": frame[:-crop_length]})
        return city_frame_list



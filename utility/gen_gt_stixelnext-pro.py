import os
import yaml
import torch
import numpy as np
import pandas as pd
from einops import rearrange
from typing import List, Dict, Tuple, Any, Optional


def _preparation_of_target_label(y_target: pd.DataFrame, n_obj_preds: int, img_size: Dict[str, int],
                                 depth_anchors: pd.DataFrame, d_scale: float = 50.0, shadowing: bool = True,
                                 u_scale: int = 8) -> torch.tensor:
    d_scale = d_scale * 0.1
    # img_path,x,yT,yB,class,depth: prepare data like normalization and scaling
    y_target['u'] = (y_target['u'] // u_scale).astype(int)  # u as index
    y_target['vT'] = (y_target['vT'] / img_size['height']).astype(float)  # vT
    y_target['vB'] = (y_target['vB'] / img_size['height']).astype(float)  # vB
    # inverted depth and scaled over 100 m
    # y_target['d'] = 1 - y_target['d'] / d_scale
    width = int(img_size['width'] / u_scale)

    gt_stx_mtx = np.zeros((width, n_obj_preds, 4))
    for index, stixel in y_target.iterrows():
        col = stixel['u']
        anchor, anchor_idx = _find_nearest_depth(depth_anchors[f'{col}'], stixel['d'])
        anchor_depth = (stixel['d'] - anchor) / d_scale
        # encoding: bottom point vB, top point vT, distance d, probability P
        gt_stx_mtx[col][anchor_idx] = [stixel['vB'], stixel['vT'], anchor_depth, 1]
        # adds a negative shadow to every entry (2 times) and set the probability accordingly
        if shadowing and anchor_idx >= 2 and gt_stx_mtx[col][anchor_idx - 1][3] == 0.0:
            anchor_depth_1 = (stixel['d'] - depth_anchors[f'{col}'][anchor_idx - 1]) / d_scale
            gt_stx_mtx[col][anchor_idx - 1] = [stixel['vB'], stixel['vT'], anchor_depth_1, 0.66]
            anchor_depth_2 = (stixel['d'] - depth_anchors[f'{col}'][anchor_idx - 2]) / d_scale
            gt_stx_mtx[col][anchor_idx - 2] = [stixel['vB'], stixel['vT'], anchor_depth_2, 0.25]
    # e.g. w=240 x n=12 x a=4
    # label = torch.from_numpy(gt_stx_mtx).to(torch.float32)
    label = rearrange(gt_stx_mtx, "w n a -> a n w")
    return label


def _find_nearest_depth(column_anchors: pd.DataFrame, depth):
    # Filter the column to get only values smaller or equal to the given value
    filtered_column = column_anchors[column_anchors <= depth]
    # If no such values exist, return the min val
    if filtered_column.empty:
        return depth, 0
    diff = (column_anchors - depth).abs()
    # Find the index of the minimum difference
    idx = diff.idxmin()
    # Get the nearest value using the index
    nearest_value = column_anchors.loc[idx]
    return nearest_value, idx


def test_stxlnxt(path):
    with open(path, "rb") as f:
        bytes_data = f.read()
        target_labels = np.frombuffer(bytes_data, dtype=np.float64).reshape(4, 48, 240)
    target_labels = torch.from_numpy(target_labels.copy()).to(torch.float32)
    print(target_labels.shape)


def main():
    with open('config.yaml') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    stixel_folder = "Stixel"
    image_folder = "FRONT"
    target_folder = "targets"
    img_size = {'height': 1280, 'width': 1920}
    depth_anchors = pd.read_csv(os.path.join(config['data_path'], "depth_anchors.csv"), index_col=0)
    filenames: List[str] = os.listdir(os.path.join(config['data_path'], config['phase'], image_folder))
    data_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]

    target_path = os.path.join(config['data_path'], config['phase'], target_folder)
    os.makedirs(target_path, exist_ok=True)
    first = True
    for data in data_map:
        stixel_csv: pd.DataFrame = pd.read_csv(os.path.join(config['data_path'], config['phase'], stixel_folder, os.path.basename(data) + ".csv"))
        target_label = _preparation_of_target_label(y_target=stixel_csv,
                                                    n_obj_preds=48,
                                                    img_size=img_size,
                                                    depth_anchors=depth_anchors)
        label_path = os.path.join(target_path, f"{os.path.basename(data)}.stxlnxt")
        with open(label_path, "wb") as f:
            f.write(target_label.tobytes())
        print(f"{os.path.basename(data)}.stxlnxt saved.")
        if first:
            first = False
            test_stxlnxt(label_path)
    print("Done.")


if __name__ == "__main__":
    main()

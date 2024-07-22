import os
import pandas as pd
import numpy as np
import yaml
with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def depth_clustering_per_row(folder_path):
    depth_distributions = {}  # Dictionary to store depth distributions per column

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Calculate column index (x // 8) and filter valid indices
            df['x'] = df['x'] // 8
            valid_indices = df['x'] < 240  # Ensure indices are within the 240-entry list

            # Group by column index and collect depth values
            for x, group in df[valid_indices].groupby('x'):
                depths = group['depth'].tolist()
                if x not in depth_distributions:
                    depth_distributions[x] = []
                depth_distributions[x].extend(depths)

    return depth_distributions


def main():
    path = os.path.join(config['data_path'], "waymo-od", "testing", "test")
    depths = depth_clustering_per_row(path)
    print("fin")


if __name__ == '__main__':
    main()

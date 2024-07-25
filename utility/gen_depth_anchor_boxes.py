import os
import faiss
import pandas as pd
import numpy as np
import yaml
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def depth_clustering_per_row(folder_path):
    depth_distributions = {}  # Dictionary to store depth distributions per column

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Calculate column index (x // 8) and filter valid indices
            df['u'] = df['u'] // 8
            valid_indices = df['u'] < 240  # Ensure indices are within the 240-entry list

            # Group by column index and collect depth values
            for x, group in df[valid_indices].groupby('u'):
                depths = group['d'].tolist()
                if x not in depth_distributions:
                    depth_distributions[x] = []
                depth_distributions[x].extend(depths)

    return depth_distributions


def main():
    path = os.path.join(config['data_path'], "waymo-od", config['phase'], "Stixel")
    depths = depth_clustering_per_row(path)
    # Near to the maximum distance from plc-config (50 - 2:padding left-right). Causes a target matrix of [4, 48, 240]
    num_clusters = 48

    anchors_by_col = []
    for i in range(len(depths)):
        col_depths = np.array(depths[i], dtype=np.float32)
        # K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(col_depths.reshape(-1, 1))
        centroids = np.sort(kmeans.cluster_centers_.squeeze())
        labels = kmeans.labels_
        anchors_by_col.append(np.round(centroids, 2))

    #anchors_by_col = [col for col in anchors_by_col]
    df = pd.DataFrame(anchors_by_col)
    df.columns = [str(i) for i in range(num_clusters)]
    df = df.T
    df.to_csv(os.path.join(config['data_path'], "waymo-od", config['phase'], "depth_anchors.csv"), index=True)

    """
    # Visualisation
    plt.scatter(depth_data, np.zeros_like(depth_data), c=labels, cmap='viridis')
    plt.scatter(centroids, np.zeros_like(centroids), color='red')
    plt.title('K-Means Clustering')
    plt.xlabel('Depth')
    plt.show()
    """


if __name__ == '__main__':
    main()

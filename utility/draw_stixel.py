import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_points_on_image(csv_file_path, image_path):
    # Load the points from the CSV file
    points = pd.read_csv(csv_file_path)
    # Load the image
    image = mpimg.imread(image_path)
    # Normalize the depth values to match the 'viridis' color map range
    depth_normalized = (points['depth'] - points['depth'].min()) / (points['depth'].max() - points['depth'].min())
    # Plot the image
    plt.figure(figsize=(20, 12))
    plt.imshow(image)
    # Scatter plot of the points with the normalized depth values as color reference
    # Adjust size and alpha for the points
    sc = plt.scatter(points['x'], points['y'], c=depth_normalized, cmap='viridis', alpha=0.7, s=10)
    # Adding color bar as legend for the depth
    cbar = plt.colorbar(sc)
    cbar.set_label('Depth')
    # Remove the axis for better visualization
    plt.axis('off')
    # Show the image and wait until a key is pressed
    plt.show()

# Define file paths
datapath = "C:\\Users\\marce\\Documents\\dataset"
images_path = os.path.join(datapath, "STEREO_LEFT")
targets_path = os.path.join(datapath, "targets_from_lidar")
file_map = []
# Durchlaufe alle Dateien im angegebenen Ordner
for filename in os.listdir(images_path):
    if filename.endswith('.png'):  # Prüfe, ob die Datei eine .png-Datei ist
        # Füge den Dateinamen ohne die .png-Endung der Liste hinzu
        file_map.append(os.path.splitext(filename)[0])

print(f'Found {len(file_map)}')

for file in file_map:
    csv_file_path = os.path.join(targets_path, file + ".csv")
    image_path = os.path.join(images_path, file + ".png")
    # output_image_path = 'path_to_save_annotated_image.png'
    plot_points_on_image(csv_file_path, image_path)


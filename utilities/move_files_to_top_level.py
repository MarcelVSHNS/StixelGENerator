import os
import shutil
import yaml

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
directory_path = config['raw_data_path']

# Gehe durch jeden Unterordner im Basisordner
for subdir, _, files in os.walk(directory_path):
    for file in files:
        # Überprüfe, ob die Dateiendung .4mse ist
        if file.endswith('.4mse'):
            # Pfad zur aktuellen Datei
            source_path = os.path.join(subdir, file)
            # Pfad zum Basisordner
            destination_path = os.path.join(directory_path, file)

            # Die Datei in den Basisordner verschieben
            shutil.move(source_path, destination_path)

print("Alle .4mse-Dateien wurden in den Basisordner verschoben.")
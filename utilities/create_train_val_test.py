import os
import random
import yaml
import shutil

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
directory_path = config['raw_data_path']
train_ratio = 0.7
val_ratio = 0.15
assert train_ratio + val_ratio >= 0, "Check ratio!"
test_ratio = 1.0 - train_ratio - val_ratio
train_dir = os.path.join(directory_path, 'training')
val_dir = os.path.join(directory_path, 'validation')
test_dir = os.path.join(directory_path, 'testing')

# Liste aller Dateien im Verzeichnis
all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Dateien zufällig mischen
random.shuffle(all_files)

# Anzahl der Dateien
num_files = len(all_files)

# Dateien aufteilen
train_files = all_files[:int(train_ratio * num_files)]
val_files = all_files[int(train_ratio * num_files):int((train_ratio + val_ratio) * num_files)]
test_files = all_files[int((train_ratio + val_ratio) * num_files):]

# Zielverzeichnisse erstellen, falls sie nicht existieren
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Dateien in die entsprechenden Verzeichnisse kopieren
for file in train_files:
    shutil.move(os.path.join(directory_path, file), train_dir)

for file in val_files:
    shutil.move(os.path.join(directory_path, file), val_dir)

for file in test_files:
    shutil.move(os.path.join(directory_path, file), test_dir)

print(f"Trainingsdateien: {len(train_files)}")
print(f"Validierungsdateien: {len(val_files)}")
print(f"Testdateien: {len(test_files)}")

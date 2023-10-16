import ameisedataset as ad
import glob
import yaml
import os

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
data_dir = config['raw_data_path'] + config['phase']

ameise_record_map = glob.glob(os.path.join(data_dir, '*.4mse'))
print(len(ameise_record_map))

for entry in ameise_record_map:
    try:
        infos, frames = ad.unpack_record(entry)
        print (entry + ' is okay!!')
    except:
        print(entry + " is corrupted...")
        os.remove(entry)
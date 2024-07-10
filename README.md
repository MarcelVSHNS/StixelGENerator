# Stixel World Generator from LiDAR data
This repo is designed to generate custom Stixel training data based on different datasets. And is related to:

[Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection](https://ieeexplore.ieee.org/Xplore/home.jsp). IV 2024.\
[Marcel Vosshans](https://scholar.google.de/citations?user=_dbcdr4AAAAJ&hl=en), [Omar Ait-Aider](https://scholar.google.fr/citations?user=NIdLQnUAAAAJ&hl=en), [Youcef Mezouar](https://youcef-mezouar.wixsite.com/ymezouar) and [Markus Enzweiler](https://markus-enzweiler.de/)\
University of Esslingen, UCA Sigma Clermont\
[[`Xplore`](https://ieeexplore.ieee.org/Xplore/home.jsp)]
## StixelGENerator
![Sample Stixel World by LiDAR](/docs/imgs/sample_stixel_world.png)
This repo provides the basic toolset to generate a Stixel World from LiDAR. It is used as Ground Truth for 
the [StixelNExT](https://github.com/MarcelVSHNS/StixelNExT) 2D estimator as well as for the 3D approach: [StixelNExT Pro](https://github.com/MarcelVSHNS/StixelNExT_Pro).

### Usage with Waymo or KITTI
1. Clone the repo to your local machine
2. Set up a virtual environment with `python -m venv venv` (we tested on Python 3.10) or Anaconda respectively `conda create -n StixelGEN python=3.9`. Activate with `source venv/bin/activate`/ `conda activate StixelGEN`
3. Install requirements with `pip install -r requirements.txt`/ `conda install --file requirements.txt` 
4. Configure the project: adapt your paths in the `config.yaml`-file and select the dataset with the import in `/generate.py` like:
```python
from dataloader import WaymoDataLoader as Dataset   # or KittiDataLoader
```
After that you can test the functionalities with `utility/explore.py` or run `/generate.py` to generate Stixel Worlds.

#### Output
The output is simply the image and a `.csv` with the following header:
> 'img_path', 'x', 'yT', 'yB', 'class', 'depth'

what can be easily read by Pandas with `pd.read_csv("target.csv")`.

#### KITTI Training Data
We also provide an already generated dataset, based on the public available KITTI dataset. It can be downloaded
[here](https://bwsyncandshare.kit.edu/s/FL4BDGe7FM2NjJK) (??? GB)

### Adaption to other Datasets
The repo is designed to work with adaptive dataloader, which can be handled by the import commands. 
For your own usage its necessary to write a new Dataloader for what the `dataloader/BaseLoader` and 
`dataloader/BaseData` can be used. Needed synchronised data/information are:
* The camera image
* LiDAR data
* Camera calibration data: 
  * Camera Matrix
  * Projection Matrix
  * Rectify Matrix
  * Transformation Matrix
* Context information (image size)
* OPTIONAL: Stereo Images (right camera)
* OPTIONAL: LiDAR Calibration (in case of global T Matrices)

#### Fine tuning
You can heavily increase the results with the parameters from `libraries/pcl-config.yaml`. 
Documentation for the functions are provided by the code. The los (line of sight) parameter can cause huge holes!

### Utilities
* explore: A simple workspace script to use and inspect the derived data, step by step.

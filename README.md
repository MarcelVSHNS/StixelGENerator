# Automatic Ground Truth - Documemtation will be completed until 30.6.24
This repo is designed to generate custom Stixel training data based on some Data Sets (branches).

## KITTI
t.b.d.

## Internal OPNV dataset (will be published soon)
t.b.d.

## Waymo
### Annotation Specification (vers. 1.4 - Segmentation)
The aim is to find cuts and identify possible objects to derive a low weight 3D representation - here: based on the segmentation
Basic (segmentation-based) rules are:
- Switch from ground to object: Ground-Cut
- Switch from instance-object/ object to object: Swib-Cut
- Switch from object to ground/instance-object: Top-Cut
- 
## Utilities
* draw_stixel: use the GT data to draw the in 2D on the related image
* concat_gt_annotations: every image generates a list of stixel, this script concatenates all training stixel into one file

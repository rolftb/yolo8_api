
vestidura-trabajo-vs - v2 2023-10-01 2:29pm
==============================

This dataset was exported via roboflow.com on October 1, 2023 at 8:00 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 100 images.
Vestidura are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Random exposure adjustment of between -30 and +30 percent

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -20 and +20 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically



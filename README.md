# Vision-RADAR Fusion with SceneSeg

The goal of this effort is to develop a Vision-Radar Fusion pipeline which can effectively combine raw radar detections with SceneSeg's semantic foreground object label predictions.

Typically, Vision-Radar Fusion is done in the Birds-Eye-View (BEV) space by projecting image features through implicit learned depth in neural network features. However, the challenge with this type of approach is that a BEV network trained with image data from one vehicle with a particular camera mounting height and camera pitch angle, fails to generalize robustly to other cameras - for example, a camera mounted on a Sedan vs an SUV vs a Truck. Such BEV approaches also fail to reliably generalize across camera FOVs, from wide-angle cameras to zoom-cameras. This makes it difficult to train a BEV network unless the final production-ready sensor configuration is fixed, and in such a case, the network would need to be trained and fine-tuned for each production setup.

To tackle this challenge, we will perform Vision-Radar Fusion in the image-space by back-projecting Radar point cloud detections into the camera image.

## Radar to Image projection examples

### ALTOS Radar 4D (Azimuth, Elevation, Range & Velocity) Radar Projection


### Standard 3D (Azimuth, Range & Velocity) Automotive Radar Projection

## Image-Space Fusion

Once RADAR points are projected into the image space, they can be associated with SceneSeg foreground object labels based on label overlap/intersection. Those Radar points which are associated with SceneSeg foreground object labels are strong candidates for foreground objects - allowing us to filter the raw Radar pointcloud to extract the most probable foreground object points, removing noisy detections. The filtered Radar detections can then be projected out to the 3D world coordinate space to create a BEV image which can be used for downstream processing.

## Datasets

To begin with, please download the [Prevention Dataset](https://prevention-dataset.uah.es/) and the [aiMotive Multi-Modal Dataset](https://www.kaggle.com/datasets/tamasmatuszka/aimotive-multimodal-dataset) as initial datasets which can be used to develop and test Image-Space Fusion algorithm. There are 4 key steps to developing the Image-Space Fusion Pipeline.

- Step 1: Parse raw data
- Step 2: Back-project RADAR points to front image
- Step 3: Associate RADAR points with SceneSeg labels
- Step 4: Project filtered RADAR points to 3D space and create BEV image

## References

- [aiMotive Dataset: A Multimodal Dataset for Robust Autonomous Driving with Long-Range Perception](https://arxiv.org/pdf/2211.09445)
- [The PREVENTION dataset: a novel benchmark for PREdiction of VEhicles iNTentIONs](https://prevention-dataset.uah.es/static/ThePREVENTIONdataset.pdf)
- [Fusion Point Prunning for Optimized 2D Object Detection with Radar-Camera Fusion](https://openaccess.thecvf.com/content/WACV2022/papers/Stacker_Fusion_Point_Pruning_for_Optimized_2D_Object_Detection_With_Radar-Camera_WACV_2022_paper.pdf)

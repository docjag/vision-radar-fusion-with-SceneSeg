"""
RADAR Data JSON path: /home/docjag/RADAR_fusion/aiMotive_multimodal_dataset/train/highway/20210401-073402-00.18.00-00.18.15@Jarvis/sensor/radar/F_LRR_C
RGB Camera Image Dir: /home/docjag/RADAR_fusion/aiMotive_multimodal_dataset/train/highway/20210401-073402-00.18.00-00.18.15@Jarvis/sensor/camera/F_MIDLONGRANGECAM_CL
"""

import os
import cv2
import json
import argparse
import logging
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Create Log files directory
log_filename = "/tmp/logs/process_aiMotive_data.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# Creating and configuring the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Creating Logging format
formatter = logging.Formatter("[%(asctime)s: %(name)s] %(levelname)s\t%(message)s")

# Creating file handler and setting the logging formats
file_handler = logging.FileHandler(log_filename, mode="a")
file_handler.setFormatter(formatter)

# Adding handlers into the logger
logger.addHandler(file_handler)

#### DIRECTORY HELPER FUNCTIONS ####


def create_output_subdirs(subdirs_list, output_dir):
    """
    Create subdirectories for the output directory
    Returns a dictionary having subdirectory paths
    """
    output_subdirs = {}

    for subdir in subdirs_list:
        subdir_path = os.path.join(output_dir, subdir)

        # Check or Create directory
        check_directory_exists(subdir_path)

        output_subdirs[subdir] = subdir_path

    return output_subdirs


def check_directory_exists(directory_path: str):
    """Check if a directory exists; if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info("Directory created: %s", directory_path)
    else:
        logger.info("Directory %s already exists.", directory_path)


def radar_to_image(radar_data, calibration_data):
    """
    Projects radar data points into the image plane using provided calibration data.

    Args:
        radar_data (list): A list of dictionaries, where each dictionary represents a radar detection.
                           Each dictionary should contain 'azimuth', 'elevation', and 'range' keys.
        calibration_data (dict): A dictionary containing calibration information for the radar and camera.
                                  It should contain 'F_LRR_C' and 'F_MIDLONGRANGECAM_CL' keys, with each containing
                                  'RT_body_from_sensor' for radar and camera, and camera intrinsics like 'focal_length_px',
                                  'principal_point_px', and 'distortion_coeffs'.

    Returns:
        list: A list of (u, v) image coordinates for each radar detection, where u and v are pixel coordinates.
              Returns an empty list if no radar data is provided.  Returns `None` for a radar detection if its projection
              fails.
    """

    if not radar_data:
        return []

    # Extract sensors calibration data
    cam_calib = calibration_data["F_MIDLONGRANGECAM_CL"]
    radar_calib = calibration_data["F_LRR_C"]

    ### Extract calibration data ###

    # Extract Radar to Body Transformation Extrinsic Matrix
    radar_to_body = np.array(radar_calib["RT_body_from_sensor"])

    # Extract Camera Calibration data
    body_to_camera = np.array(cam_calib["RT_body_from_sensor"])
    focal_length_px = np.array(cam_calib["focal_length_px"])
    principal_point_px = np.array(cam_calib["principal_point_px"])
    distortion_coeffs = np.array(cam_calib["distortion_coeffs"])
    image_resolution_px = cam_calib["image_resolution_px"]

    # Compose the transformations: radar -> body -> camera
    radar_to_camera = body_to_camera @ radar_to_body

    image_coords = []

    for detection in radar_data:
        azimuth = detection["azimuth"]
        elevation = detection["elevation"]
        range_val = detection["range"]

        # Convert spherical coordinates to Cartesian coordinates in radar frame
        x_radar = range_val * np.cos(elevation) * np.cos(azimuth)
        y_radar = range_val * np.cos(elevation) * np.sin(azimuth)
        z_radar = range_val * np.sin(elevation)

        # Transform radar point to camera frame
        radar_point_homogeneous = np.array([x_radar, y_radar, z_radar, 1.0])
        camera_point_homogeneous = radar_to_camera @ radar_point_homogeneous
        camera_point = camera_point_homogeneous[:3]  # Extract x, y, z

        # Project to image plane
        x_image = focal_length_px[0] * (camera_point[0] / camera_point[2])
        y_image = focal_length_px[1] * (camera_point[1] / camera_point[2])

        # Apply principal point offset
        u = x_image + principal_point_px[0]
        v = y_image + principal_point_px[1]

        # Apply distortion (Brown-Conrady model)
        x = (u - principal_point_px[0]) / focal_length_px[0]
        y = (v - principal_point_px[1]) / focal_length_px[1]
        r2 = x * x + y * y
        radial_distortion = (
            1
            + distortion_coeffs[0] * r2
            + distortion_coeffs[1] * r2 * r2
            + distortion_coeffs[4] * r2 * r2 * r2
        )
        x_distorted = (
            x * radial_distortion
            + 2 * distortion_coeffs[2] * x * y
            + distortion_coeffs[3] * (r2 + 2 * x * x)
        )
        y_distorted = (
            y * radial_distortion
            + 2 * distortion_coeffs[3] * x * y
            + distortion_coeffs[2] * (r2 + 2 * y * y)
        )
        u = focal_length_px[0] * x_distorted + principal_point_px[0]
        v = focal_length_px[1] * y_distorted + principal_point_px[1]

        # Check if projected point is within image bounds. Return None if out of bounds
        if not (0 <= u < image_resolution_px[0] and 0 <= v < image_resolution_px[1]):
            image_coords.append(None)
        else:
            image_coords.append((u, v))

    return image_coords


def main(args):
    radar_json = "/home/docjag/python_garage/vision_radar_fusion/F_LRR_C_0016465.json"
    calib_json = "/home/docjag/python_garage/vision_radar_fusion/calibration.json"
    image_path = "/home/docjag/python_garage/vision_radar_fusion/F_MIDLONGRANGECAM_CL_0016465.jpg"

    with open(radar_json, "r") as fh:
        radar_data = json.load(fh)

    obj_id = radar_data["id"]
    targets = radar_data["targets"]

    print(len(targets))
    print(type(targets))

    with open(calib_json, "r") as cfh:
        calib_data = json.load(cfh)

    # Extract sensors calibration data
    cam_calib = calib_data["F_MIDLONGRANGECAM_CL"]
    radar_calib = calib_data["F_LRR_C"]

    # Extract Radar to Body Transformation Extrinsic Matrix
    radar2body_transform = radar_calib["RT_body_from_sensor"]
    radar2body_transform = np.array(radar2body_transform)
    print(type(radar2body_transform))

    # Load the camera image (replace with your actual image path)
    image_path = "path_to_your_camera_image.jpg"
    image = cv2.imread(image_path)

    # Process each radar point and project it onto the image
    projected_points = []
    for point in [radar_point]:  # Replace with your actual radar points list
        projected = radar_to_camera_point(point, radar_to_cam_transform, camera_matrix)
        if projected:
            projected_points.append(projected)

    # Draw the projected points on the image
    for x, y in projected_points:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Display the output
    cv2.imshow("Radar Points on Camera Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process aiMotive dataset - for Vision-RADAR Fusion"
    )
    parser.add_argument(
        "--image-dir",
        "-i",
        type=str,
        required=True,
        help="""
        aiMotive Image Datasets directory. 
        DO NOT include subdirectories or files.""",
    )
    parser.add_argument(
        "--radar-dir",
        "-r",
        type=str,
        required=True,
        help="""
        aiMotive RADAR JSON files directory.
        Do not include subdirectories or files.""",
    )

    parser.add_argument(
        "--calibration-dir",
        "-c",
        type=str,
        required=True,
        help="""
        aiMotive sensor calibration JSON files directory.
        Do not include subdirectories or files.""",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for back project images",
    )
    args = parser.parse_args()

    main(args)

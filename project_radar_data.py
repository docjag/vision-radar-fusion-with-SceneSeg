"""
@Author: Sohel Mahmud
@Date:
@Description: Vision RADAR Fusion for SceneSeg
"""

import os
import glob
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


def check_directory_exists(directory_path: str):
    """Check if a directory exists; if not, create it."""

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info("Directory created: %s", directory_path)

    else:
        logger.info("Directory %s already exists.", directory_path)


def spherical_to_cartesian(azimuth, elevation, range):
    """
    Convert spherical coordinates to 3D Cartesian coordinates.
    """
    # Transform Radial to Cartesian Coordiates
    x = range * np.cos(elevation) * np.cos(azimuth)
    y = range * np.cos(elevation) * np.sin(azimuth) * -1.0
    z = range * np.sin(elevation)

    if z < -0.45:
        z = 0

    return np.array([x, y, z, 1.0])


def get_circle_radius(range_val):

    index = int((range_val - 1) // 30)
    return 5 - index


def read_json(json_file):
    with open(json_file, "r") as fh:
        json_data = json.load(fh)

    return json_data


def write_radar_detections_overlay(image_path, metadata, output_dir):
    """Create RADAR Detections Overlay"""

    # Read Image File
    img = cv2.imread(image_path)

    # Extract Image Name
    image_name = os.path.basename(image_path)

    for i in metadata:
        x = i["u"]
        y = i["v"]

        range_value = i["range_value"]

        # Get Circle Radius based on Range Value
        circle_rad = get_circle_radius(range_value)
        print(f"Circle radius: {circle_rad}")

        # Draw Circles for the Detected Objects
        cv2.circle(img, (x, y), circle_rad, (0, 255, 0), 1)

    # Save the Output Images
    cv2.imwrite(os.path.join(output_dir, image_name), img=img)

    # # Display the output
    # cv2.imshow("Radar Points on Camera Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_frameID(json_file):
    """Extract Frame ID from the File Name"""
    # Extract Frame ID
    frame_id = os.path.basename(json_file).split("_")[-1]

    # Exclude .json part from frame_id
    frame_id = frame_id.split(".")[0]

    return frame_id


def project_all_radar_detections(P, sensors_dict, drive, output_dir):
    """Project All RADAR Detections onto All Camera Frames"""

    # RADAR Detection JSON Files Path for Each Drive
    radar_jsons_path = os.path.join(drive, "radar", sensors_dict["radar"])

    # Set CAMERA Image Directory
    img_dir = os.path.join(drive, "camera", sensors_dict["camera"])

    for radar_json in glob.glob(f"{radar_jsons_path}/*.json"):

        # Project Individual RADAR Detections on Image Plane
        # Get the Projection Meta data: (x, y) coordinates, range and speed as a List
        projection_metadata = project_radar2image_plane(P, radar_json)

        # Extract Camera Name
        camera_name = sensors_dict["camera"]

        # Extract Frame ID & metadata
        frame_id = projection_metadata["id"]
        metadata = projection_metadata["metadata"]

        # Process Image path
        img_path = os.path.join(img_dir, f"{camera_name}_{frame_id}.jpg")

        # Write the RADAR Detections Overlay
        write_radar_detections_overlay(img_path, metadata, output_dir)


def project_radar2image_plane(P_full, radar_json):
    """Project RADAR detection for a Single Camera Frame"""

    # Read Calibration JSON file
    # Read RADAR Targets Data as List of dictionary
    radar_detections = read_json(radar_json)["targets"]

    # Extract Frame ID from the File Name
    frame_id = get_frameID(json_file=radar_json)

    projection_metadata = []

    for detection in radar_detections:

        # Process RADAR data
        azimuth = detection["azimuth"]
        elevation = detection["elevation"]
        range_val = detection["range"]
        speed_val = detection["speed"]

        # Convert to 3D RADAR Homogenous Point in Cartesian coordinates (4D Vector)
        pt_radar_homogenous = spherical_to_cartesian(azimuth, elevation, range_val)

        pt_img = P_full @ pt_radar_homogenous
        print(f"Image Frame: {pt_img}")

        u, v, w = pt_img

        # Convert to 2D Image Coordinate from 3D Homogeneous Coordinate
        u = int(u / w)
        v = int(v / w)
        print(f"Image coor: {u} {v}")

        projection_metadata.append(
            {
                "u": u,
                "v": v,
                "range_value": range_val,
                "radial_speed": speed_val,
            }
        )

    return {"id": frame_id, "metadata": projection_metadata}


def calculate_full_projection_matrix(calib_file_path, sensors_dict):
    """Calculate Full Projection Matrix for each Drive"""

    #################################################
    ########## CALIBRATION DATA EXTRACTION ##########
    #################################################

    # Read Calibration JSON file
    calib_data = read_json(calib_file_path)

    ###################################
    ####### SENSOR EXINTRINSICS #######
    ###################################

    ### RADAR Calibration ###
    # Extract RADAR Calibration data
    radar_calib = calib_data[sensors_dict["radar"]]

    # Extract Camera Calibration data
    cam_calib = calib_data[sensors_dict["camera"]]

    # Extract Radar to Body Transformation Matrix (4x4 matrix)
    T_radar_to_body = np.array(radar_calib["RT_body_from_sensor"])

    # Vehicle Body to Camera Transformation Matrix (4x4 matrix)
    T_body_to_camera = np.array(cam_calib["RT_sensor_from_body"])

    # Complete transformation from RADAR to Camera
    T_radar_to_cam = T_body_to_camera @ T_radar_to_body

    #################################
    ####### CAMERA INTRINSICS #######
    #################################

    # Camera Intrinsic Parameters
    f_x, f_y = cam_calib["focal_length_px"]  # Focal length in Pixels
    c_x, c_y = cam_calib["principal_point_px"]  # Principal Point in Pixels

    # Create Camera Intrinsic Matrix (3x4 matrix)
    K = np.array([[f_x, 0.0, c_x, 0.0], [0.0, f_y, c_y, 0.0], [0.0, 0.0, 1.0, 0.0]])

    ######################################
    ####### FULL PROJECTION MATRIX #######
    ######################################
    # Full Projection Matrix (Intrinsic + Extrinsics) maps 3D RADAR Detections to 2D image points
    P = K @ T_radar_to_cam

    return P


def main(args):
    """Main function"""

    # Unpack Command line arguments
    dataset = args.dataset
    driving_condition = args.driving_conditions
    output_dir = args.output_dir

    # Check the output directory exists or Create a new one
    check_directory_exists(output_dir)

    # Create Sensors Metadata
    sensors_dict = {
        "calibration": "calibration.json",
        "camera": "F_MIDLONGRANGECAM_CL",
        "radar": "F_LRR_C",
    }

    # Drives for Individual Condition (ex, highway)
    drives_dir = os.path.join(dataset, driving_condition)

    # Iterate through each drive
    for drive in glob.glob(f"{drives_dir}/*/sensor"):
        # Calibration File for Each Drive
        calib_json = os.path.join(drive, "calibration", sensors_dict["calibration"])

        # Calculate the Full Projection Matrix
        P = calculate_full_projection_matrix(
            calib_file_path=calib_json, sensors_dict=sensors_dict
        )

        project_all_radar_detections(P, sensors_dict, drive, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process aiMotive dataset - for Vision-RADAR Fusion"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="""
        aiMotive RADAR JSON files directory.
        Do not include subdirectories or files.""",
    )

    parser.add_argument(
        "--driving-conditions",
        "-c",
        type=str,
        required=True,
        help="""
        Driving Conditions: [highway,  night, rain, urban] 
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="~/tmp/radar_output",
        help="Output directory for back project images",
    )
    args = parser.parse_args()

    main(args)

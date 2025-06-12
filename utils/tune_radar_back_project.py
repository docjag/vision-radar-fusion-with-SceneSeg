"""
@Author: Sohel Mahmud
@Date:
@Description: Vision RADAR Fusion for SceneSeg
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def spherical_to_cartesian(azimuth, elevation, range):
    """
    Convert spherical coordinates to 3D Cartesian coordinates.
    """
    # Transform Radial to Cartesian Coordiates
    x = range * np.cos(elevation) * np.cos(azimuth)
    y = range * np.cos(elevation) * np.sin(azimuth) * -1.0
    z = range * np.sin(elevation)

    if z < -1.0:
        z = 0

    return np.array([x, y, z, 1.0])


def read_json(json_file):
    with open(json_file, "r") as fh:
        json_data = json.load(fh)

    return json_data


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


def generate_radar_meshgrid():
    """Generate Sample RADAR Points in XY Plane"""

    # Generate Sample RADAR Points in XY Plane
    y_coords = np.arange(-20, 21, 5)
    x_coords = np.arange(60, 81, 5)

    # Create a meshgrid for the coordinates
    Y, X = np.meshgrid(y_coords, x_coords)

    # Flatten the grid for plotting points
    points_x = X.flatten()
    points_y = Y.flatten()

    return points_x, points_y


def create_fake_pc(y):
    z = 0
    x = 55
    point_clouds = []
    for i in range(1, 10):
        x += 5
        point_clouds.append([x, y, z, 1.0])

    return np.array(point_clouds)


def visualize_radar_meshgrid(point_cloud):
    """Visualize Sample RADAR Points in XY Plane"""

    plt.figure(figsize=(8, 8))
    plt.scatter(
        point_cloud[:, 1], point_cloud[:, 0], c="red", s=20, label="Sample RADAR Points"
    )
    plt.ylabel("X (Longitudinal) (m)")
    plt.xlabel("Y (Lateral) (m)")
    plt.title("Sample RADAR Points in XY Plane")
    plt.grid()
    plt.axis("equal")
    plt.show()


def visualize_radar2image_points(projected_points):
    """Visualize Projected RADAR Points onto Image Plane"""
    plt.figure(figsize=(8, 8))
    plt.scatter(
        projected_points[:, 0],
        projected_points[:, 1],
        c="red",
        s=20,
        label="Projected RADAR Points",
    )
    plt.xlabel("Image X Coordinate (u)")
    plt.ylabel("Image Y Coordinate (v)")
    plt.title("Projected RADAR Points onto Image Plane")
    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.show()


def main(args):

    # calib_json = "sensor/calibration/calibration.json"
    calib_json = args.calib_json

    # Create Sensors Metadata
    sensors_dict = {
        "calibration": "calibration.json",
        "camera": "F_MIDLONGRANGECAM_CL",
        "radar": "F_LRR_C",
    }

    # Calculate the Full Projection Matrix
    P = calculate_full_projection_matrix(
        calib_file_path=calib_json, sensors_dict=sensors_dict
    )

    # Generate Sample RADAR Points in XY Plane
    point_cloud = np.concatenate(
        (
            create_fake_pc(-10),
            create_fake_pc(-5),
            create_fake_pc(0),
            create_fake_pc(5),
            create_fake_pc(10),
        ),
        axis=0,
    )

    # Visualize Sample RADAR Points in XY Plane
    visualize_radar_meshgrid(point_cloud)

    # Project RADAR Points onto Image Plane using the Full Projection Matrix
    projected_points = (P @ point_cloud.T).T
    projected_points = (projected_points[:, :2] / projected_points[:, 2:3]).astype(int)

    # Visualize Projected RADAR Points onto Image Plane
    visualize_radar2image_points(projected_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process aiMotive dataset - for Vision-RADAR Fusion"
    )

    parser.add_argument(
        "--calib-json",
        "-c",
        type=str,
        required=True,
        help="""
        Calibration JSON file path. Example: --calib-json sensor/calibration/calibration.json
        This file contains the calibration data for the camera and radar sensors.
        It should include the transformation matrices and intrinsic parameters for both sensors.
        Example: sensor/calibration/calibration.json
        """,
    )

    args = parser.parse_args()

    main(args)

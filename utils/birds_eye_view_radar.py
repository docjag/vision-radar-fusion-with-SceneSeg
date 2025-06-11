"""
@Author: Sohel Mahmud
@Date:
@Description: Vision RADAR Fusion for SceneSeg
"""

import os
import argparse
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def get_circle_radius(range_val):

    index = int((range_val - 1) // 30)
    return 5 - index


def read_json(json_file):
    with open(json_file, "r") as fh:
        json_data = json.load(fh)

    return json_data


def draw_text(image, text, position, offset=(0, 0)):
    """
    Draw text on the image at a specified position with an optional offset.
    """
    cv2.putText(
        image,
        text,
        (position[0] + offset[0], position[1] + offset[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.25,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def display_image(image):
    cv2.imshow("Bird's Eye View", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_min_vals(points):
    min_x = min(x for x, y in points)
    min_y = min(y for x, y in points)

    return min_x, min_y


def convert_angular_to_cartesianXY(azimuth, elevation, range):
    """
    Convert spherical coordinates to 3D Cartesian coordinates.
    """
    # Transform Radial to Cartesian Coordiates
    x = range * np.cos(elevation) * np.cos(azimuth)
    y = range * np.cos(elevation) * np.sin(azimuth) * -1.0
    z = range * np.sin(elevation)

    if z < -0.45:
        z = 0

    return x, y, z


def get_fov_points(
    radar_detections,
    max_range,
    fov_angle_deg=32,
):
    """Get points for the Field of View (FOV) lines."""
    # Example FOV angles in degrees
    fov_start_angle_deg = -fov_angle_deg / 2
    fov_end_angle_deg = fov_angle_deg / 2

    # Convert angles to radians for calculations
    fov_start_angle_rad = np.deg2rad(fov_start_angle_deg)
    fov_end_angle_rad = np.deg2rad(fov_end_angle_deg)

    # Calculate end points for the FOV lines
    x_start = max_range * np.cos(fov_start_angle_rad)
    y_start = max_range * np.sin(fov_start_angle_rad)

    x_end = max_range * np.cos(fov_end_angle_rad)
    y_end = max_range * np.sin(fov_end_angle_rad)

    return (x_start, y_start), (x_end, y_end)


def visualize_radar_bev(visual_meta):
    """Visualize RADAR detections in Bird's Eye View (BEV) with speed color coding."""

    x_coords = visual_meta["x_coords"]
    y_coords = visual_meta["y_coords"]
    speeds = visual_meta["speeds"]
    fov_start = visual_meta["fov_start"]
    fov_end = visual_meta["fov_end"]
    range_rings = visual_meta["range_rings"]
    min_speed = visual_meta["min_speed"]
    max_speed = visual_meta["max_speed"]

    # Unpack FOV start and end points
    x_start, y_start = fov_start
    x_end, y_end = fov_end

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Swap axes as before
    sc = ax.scatter(y_coords, x_coords, c=speeds, s=25, marker="o", cmap="viridis")

    # Set Axis labels and titles
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title(f"Bird's eye view - RADAR Detections (FOV: -16 to 16 degrees)")
    ax.grid(True)
    ax.invert_xaxis()

    #####################
    #### RANGE RINGS ####
    #####################
    for r in range_rings[1:]:
        circle = plt.Circle(
            (0, 0), r, edgecolor="gray", facecolor="none", linestyle="--", alpha=0.5
        )
        ax.add_artist(circle)

    #################
    ### FOV LINES ###
    #################
    # Plot FOV lines
    ax.plot([0, y_start], [0, x_start], color="blue", linestyle="--", linewidth=1)
    ax.plot([0, y_end], [0, x_end], color="blue", linestyle="--", linewidth=1)

    ########################################
    ### Colorbar for Speed Visualization ###
    ########################################
    # Add a colorbar to show speed scale
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Speed (m/s)")

    # Define tick positions at min, max, and optionally intermediate speeds
    tick_positions = np.linspace(min_speed, max_speed, num=5)
    tick_labels = [f"{speed:.2f}" for speed in tick_positions]

    # Set the ticks and labels on the colorbar
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    plt.axis("equal")
    plt.show()


def process_radar_coordinates(radar_file_path):
    """Project RADAR detection for a Single Camera Frame"""

    # Read Calibration JSON file
    # Read RADAR Targets Data as List of dictionary
    radar_detections = read_json(radar_file_path)["targets"]

    x_coords = []
    y_coords = []
    ranges = []
    speeds = []

    visual_meta = {}

    # Initialize lists to store coordinates, ranges, and speeds
    for indx, detection in enumerate(radar_detections):

        # Process RADAR data
        azimuth = detection["azimuth"]
        elevation = detection["elevation"]
        range_val = detection["range"]
        speed_val = detection["speed"]

        # Convert to 3D RADAR Homogenous Point in Cartesian coordinates (4D Vector)
        x, y, _ = convert_angular_to_cartesianXY(azimuth, elevation, range_val)

        x_coords.append(x)
        y_coords.append(y)

        ranges.append(range_val)
        speeds.append(speed_val)

    # Compute maximum range once
    max_range = max(ranges)

    # Generate points to draw the FOV lines
    fov_p1, fov_p2 = get_fov_points(radar_detections, max_range)

    # Compute max and min speeds
    max_speed, min_speed = max(speeds), min(speeds)

    print(f"Max Speed: {max_speed}")
    print(f"Min Speed: {min_speed}")

    # Generate range rings with numpy for efficiency
    range_rings = np.linspace(0, max_range, 5)

    visual_meta = {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "fov_start": fov_p1,
        "fov_end": fov_p2,
        "max_range": max_range,
        "range_rings": range_rings,
        "speeds": speeds,
        "min_speed": min_speed,
        "max_speed": max_speed,
    }

    return visual_meta


def main(args):
    frame_id = args.frame_id

    # Set RADAR JSON File path
    radar_json = f"sensor/radar/F_LRR_C/F_LRR_C_00{frame_id}.json"

    # Convert RADAR detections to RADAR Coordinates (in meter)
    visual_meta_data = process_radar_coordinates(radar_json)
    visualize_radar_bev(visual_meta_data)

    # print("Visual Metadata:", visual_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process aiMotive dataset - for Vision-RADAR Fusion"
    )

    parser.add_argument(
        "--frame-id",
        "-f",
        type=str,
        required=True,
        help="5-digit Frame ID without trailing zeros",
    )

    args = parser.parse_args()

    main(args)

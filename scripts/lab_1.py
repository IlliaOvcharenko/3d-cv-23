"""
Se-up description:
- 8×8 squares, 256 corners
- the size of the pattern is 17cm x 17cm
- the 2D coordinates (in inches) of these points are available here.
  (we assume the plane is at z=0)
- 
- the format of the calibration file is: a, c, b, u0, v0, k1, k2, 
  then the rotation matrix and translation vector for the first image
  the rotation matrix and translation vector for the second image, etc.
"""

import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import math
import numpy as np
import matplotlib.pyplot as plt

from typing import (Dict)
from pathlib import Path
from fire import Fire
from more_itertools import pairwise
from scipy.spatial.transform import Rotation


def read_camera_calibration_matrix(fn: Path) -> Dict:
    _read_line_of_floats = lambda l: list(map(float, l.split(" ")))

    with open(fn, "r") as f:
        lines = f.readlines()
    a, c, b, u_0, v_0  = _read_line_of_floats(lines[0])
    k_1, k_2 = _read_line_of_floats(lines[2])


    images = []

    for image_idx in range(5):
        r_lines_idx = range(4 + (image_idx * 5), 7 + (image_idx * 5))
        t_line_idx = 7 + (image_idx * 5)
        r = np.array([_read_line_of_floats(lines[l]) for l in r_lines_idx]).T
        t = np.array([_read_line_of_floats(lines[t_line_idx])]).T

        rt = np.vstack([np.hstack([r, t]), np.array([[0, 0, 0, 1]])])
        images.append({"r": r, "t": t, "rt": rt})
    return {
        "a": a,
        "c": c,
        "b": b,
        "u_0": u_0,
        "v_0": v_0,
        "k_1": k_1,
        "k_2": k_2,

        "images": images,
    }


def homogeneous_to_euclidian(v):
    return v[:3, :] / v[3, 0]

def euclidian_to_homogeneous(v):
    return np.vstack([v, np.array([1])])


def draw_fov(ax, camera_origin, fov_points, fov_color):
    for fov_p in fov_points:
        ax.plot(
            [camera_origin[0], fov_p[0]],
            [camera_origin[1], fov_p[1]],
            [camera_origin[2], fov_p[2]],
            color=fov_color,
        )

    for (fov_p1, fov_p2) in pairwise(fov_points + [fov_points[0]]):
        ax.plot(
            [fov_p1[0], fov_p2[0]],
            [fov_p1[1], fov_p2[1]],
            [fov_p1[2], fov_p2[2]],
            color=fov_color,
        )


def draw_basis(ax, origin, x, y, z, name, fov_points=None, fov_color=None):


    ax.plot(
        [origin[0], x[0]],
        [origin[1], x[1]],
        [origin[2], x[2]],
        color="red",
        label="x axis",
    )
    ax.plot(
        [origin[0], y[0]],
        [origin[1], y[1]],
        [origin[2], y[2]],
        color="blue",
        label="y axis",
    )

    ax.plot(
        [origin[0], z[0]],
        [origin[1], z[1]],
        [origin[2], z[2]],
        color="green",
        label="z axis",
    )
    origin_text_shift = np.array([[-0.2, -0.2, -0.2]]).T
    ax.text(*(origin + origin_text_shift).T[0], name)
    ax.text(*x.T[0], f"x")
    ax.text(*y.T[0], f"y")
    ax.text(*z.T[0], f"z")

    if fov_points is not None:
        draw_fov(ax, origin, fov_points, fov_color)


def draw_board(ax, p1, p2, p3, p4, color="magenta"):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color=color,
    )
    ax.plot(
        [p2[0], p3[0]],
        [p2[1], p3[1]],
        [p2[2], p3[2]],
        color=color,
    )
    ax.plot(
        [p3[0], p4[0]],
        [p3[1], p4[1]],
        [p3[2], p4[2]],
        color=color,
    )
    ax.plot(
        [p4[0], p1[0]],
        [p4[1], p1[1]],
        [p4[2], p1[2]],
        color=color,
    )


def calculate_fov_points(calibration, z_axis, fov_scale):
    fx = calibration["a"]
    fy = calibration["b"]
    fov_x = math.degrees(math.atan(640 / (2*fx)))
    fov_y = math.degrees(math.atan(480 / (2*fy)))
    # print(fx, fy)
    # print(fov_x, fov_y)

    fov_vec = z_axis * fov_scale
    fov_points = [
        Rotation.from_euler("xy", [fov_y, fov_x], degrees=True).as_matrix() @ fov_vec,
        Rotation.from_euler("xy", [-fov_y, fov_x], degrees=True).as_matrix() @ fov_vec,
        Rotation.from_euler("xy", [-fov_y, -fov_x], degrees=True).as_matrix() @ fov_vec,
        Rotation.from_euler("xy", [fov_y, -fov_x], degrees=True).as_matrix() @ fov_vec,
    ]
    return fov_points

def board_centric(calibration):
    board_origin_world = np.array([[0, 0, 0]]).T
    board_z_axis_world = np.array([[0, 0, 1]]).T
    board_y_axis_world = np.array([[0, 1, 0]]).T
    board_x_axis_world = np.array([[1, 0, 0]]).T

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    # board_shape = (170, 170) #mm units
    # board_shape = (6.69291, 6.69291) #inches units
    board_shape = (17, 17)

    board_p1 = board_origin_world
    board_p2 = board_p1 + np.array([[board_shape[0], 0, 0]]).T
    board_p3 = board_p2 + np.array([[0, board_shape[1], 0]]).T
    board_p4 = board_p1 + np.array([[0, board_shape[1], 0]]).T

    draw_board(
        ax,
        board_p1,
        board_p2,
        board_p3,
        board_p4,
    )

    draw_basis(
        ax,
        board_origin_world,
        board_x_axis_world,
        board_y_axis_world,
        board_z_axis_world,
        "Board location",
    )

    for image_idx in range(len(calibration["images"])):
        from_world_to_camera = calibration["images"][image_idx]["rt"]
        from_camera_to_world = np.linalg.inv(from_world_to_camera)

        origin_camera = np.array([[0, 0, 0, 1]]).T
        z_axis_camera = np.array([[0, 0, 1, 1]]).T
        y_axis_camera = np.array([[0, 1, 0, 1]]).T
        x_axis_camera = np.array([[1, 0, 0, 1]]).T


        origin_world = from_camera_to_world @ origin_camera
        origin_world = homogeneous_to_euclidian(origin_world)

        z_axis_world = from_camera_to_world @ z_axis_camera
        z_axis_world = homogeneous_to_euclidian(z_axis_world)

        y_axis_world = from_camera_to_world @ y_axis_camera
        y_axis_world = homogeneous_to_euclidian(y_axis_world)

        x_axis_world = from_camera_to_world @ x_axis_camera
        x_axis_world = homogeneous_to_euclidian(x_axis_world)

        fov_color = "lightblue"
        fov_scale = 5
        fov_points = calculate_fov_points(calibration,
                                          homogeneous_to_euclidian(z_axis_camera),
                                          fov_scale)

        fov_points = [from_camera_to_world @ euclidian_to_homogeneous(fp) for fp in fov_points]
        fov_points = [homogeneous_to_euclidian(fp) for fp in fov_points]

        # Plot camera location for image 1
        draw_basis(
            ax,
            origin_world,
            x_axis_world,
            y_axis_world,
            z_axis_world,
            f"Cam {image_idx+1}",

            fov_points=fov_points,
            fov_color=fov_color,
        )

    # ax.set_aspect("equal")
    # ax.legend()
    # ax.view_init(elev=40, azim=-20, roll=0)
    ax.view_init(elev=0, azim=-10, roll=180, vertical_axis="z")
    plt.title("Board Centric Coordinates")

    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    # ax.set_zlim([0, 20])
    # plt.show()

    plt.savefig("board-centric.png", bbox_inches="tight")


def camera_centric(calibration):

    camera_origin = np.array([[0, 0, 0]]).T
    camera_z_axis = np.array([[0, 0, 1]]).T
    camera_y_axis = np.array([[0, 1, 0]]).T
    camera_x_axis = np.array([[1, 0, 0]]).T

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    fov_color = "lightblue"
    fov_scale = 5
    fov_points = calculate_fov_points(calibration, camera_z_axis, fov_scale)

    draw_basis(
        ax,
        camera_origin,
        camera_x_axis,
        camera_y_axis,
        camera_z_axis,
        "Camera location",

        fov_points=fov_points,
        fov_color=fov_color,
    )



    # board_shape = (170, 170) #mm units
    # board_shape = (6.69291, 6.69291) #inches units
    board_shape = (17, 17)

    board_colors = ["magenta", "orange", "olive", "teal", "indigo"]
    for image_idx in range(len(calibration["images"])):
        from_world_to_camera = calibration["images"][image_idx]["rt"]
        # from_camera_to_world = np.linalg.inv(rt)

        board_p1 = np.array([[0, 0, 0, 1]]).T
        board_p2 = board_p1 + np.array([[board_shape[0], 0, 0, 0]]).T
        board_p3 = board_p2 + np.array([[0, board_shape[1], 0, 0]]).T
        board_p4 = board_p1 + np.array([[0, board_shape[1], 0, 0]]).T

        board_p1_camera = from_world_to_camera @ board_p1
        board_p1_camera = homogeneous_to_euclidian(board_p1_camera)

        board_p2_camera = from_world_to_camera @ board_p2
        board_p2_camera = homogeneous_to_euclidian(board_p2_camera)

        board_p3_camera = from_world_to_camera @ board_p3
        board_p3_camera = homogeneous_to_euclidian(board_p3_camera)

        board_p4_camera = from_world_to_camera @ board_p4
        board_p4_camera = homogeneous_to_euclidian(board_p4_camera)

        draw_board(
            ax,
            board_p1_camera,
            board_p2_camera,
            board_p3_camera,
            board_p4_camera,
            board_colors[image_idx],
        )
        ax.text(*board_p1_camera.T[0], f"{image_idx+1}")


        # origin_camera = np.array([[0, 0, 0, 1]]).T
        # z_axis_camera = np.array([[0, 0, 1, 1]]).T
        # y_axis_camera = np.array([[0, 1, 0, 1]]).T
        # x_axis_camera = np.array([[1, 0, 0, 1]]).T
    ax.view_init(elev=20, azim=-120, roll=0, vertical_axis="y")
    # ax.view_init(elev=0, azim=-20, roll=-90)
    plt.title("Camera Centric Coordinates")
    plt.savefig("camera-centric.png", bbox_inches="tight")
    # plt.savefig("test.png", bbox_inches="tight")

    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    # ax.set_zlim([0, 20])
    # plt.show()



def main():
    data_folder = Path("data")
    calibration_fn = data_folder / "completecalibration.txt"
    calibration = read_camera_calibration_matrix(calibration_fn)

    # TODO add axis limits
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    # ax.set_zlim([0, 20])

    camera_centric(calibration)
    board_centric(calibration)


if __name__ == "__main__":
    Fire(main)

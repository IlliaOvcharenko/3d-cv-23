import sys, os
sys.path.append(os.getcwd())

# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from typing import (Dict)
from pathlib import Path
from fire import Fire

from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer



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
        r = np.array([_read_line_of_floats(lines[l]) for l in r_lines_idx])
        # r = np.array([_read_line_of_floats(lines[l]) for l in r_lines_idx]).T
        t = np.array([_read_line_of_floats(lines[t_line_idx])])
        # print(r, t)

        images.append({"r": r, "t": t})

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


def main():
    data_folder = Path("data")
    calibration_fn = data_folder / "completecalibration.txt"
    calibration = read_camera_calibration_matrix(calibration_fn)

    visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 100])


    board_origin_world = np.array([[0, 0, 0]]).T

    # board_shape = (170, 170) #mm units
    board_shape = (6.69291, 6.69291) #inches units

    board_p1 = board_origin_world
    board_p2 = board_p1 + np.array([[board_shape[0], 0, 0]]).T
    board_p3 = board_p2 + np.array([[0, board_shape[1], 0]]).T
    board_p4 = board_p1 + np.array([[0, board_shape[1], 0]]).T

    draw_board(
        visualizer.ax,
        board_p1,
        board_p2,
        board_p3,
        board_p4,
    )

    for image_idx in range(len(calibration["images"])):
        r = calibration["images"][image_idx]["r"]
        t = calibration["images"][image_idx]["t"]
        # print(r.shape, t.shape)
        rt = np.vstack([np.hstack([r.T, t.T]), np.array([[0, 0, 0, 1]])])
        visualizer.extrinsic2pyramid(rt, 'c', 10)
        break
    visualizer.show()

    # visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)
    # visualizer.show()

if __name__ == "__main__":
    Fire(main)

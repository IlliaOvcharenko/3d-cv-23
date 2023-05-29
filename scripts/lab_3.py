"""
- [x] Use board #3 for this lab.

- [x] Undistort the detected corners as in Lab 02.

- [x] Join all points in the rows and columns to get two sets of imaged parallel lines.

- [x] Compute the meets of imaged parallel lines (the vanishing points)

- [x] Plot the corners, joins and meets in the image.

- [x] Normalize the image points using the given principal point and skew from Zhang

- [ ] Compute the focal length from the two vanishing points using the fact that they are orthogonal in the scene.

- [ ] Compute the focal length in an auto-calibration context, meaning, make the assumption that there is zero skew and that principal point is at the enter of the image.

- [ ] Compute the relative error of the focal length with respect to the gold-standard calibration computed by Zhang of result from (6) and (7).

- [ ] Ortho-rectify the calibration board using the two vanishing points.
"""
import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import cv2
import scipy

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import (Dict, Tuple, List)
from pathlib import Path
from fire import Fire


def read_fiducial_points(jean_yves_folder: Path) -> Dict:
    def _read_points_from_file(fn: Path) -> npt.NDArray:
        with open(fn, "r") as f:
            lines = f.readlines()
        boxes = [l.strip().split("    ") for l in lines]
        coords = [coords.split(" ") for b in boxes for coords in b]
        coords = [list(map(float, coord)) for coord in coords]
        coords = np.array(coords)
        return coords.T

    images = []
    for img_idx in range(1, 5+1):
        fn = jean_yves_folder / f"data{img_idx}.txt"
        images.append({"distorted": _read_points_from_file(fn)})

    board_fn = jean_yves_folder / f"Model.txt"
    board = _read_points_from_file(board_fn)

    points = {
        "images": images,
        "board": board,
    }
    return points


def read_camera_calibration_matrix(fn: Path) -> Dict:
    _read_line_of_floats = lambda l: list(map(float, l.split(" ")))

    with open(fn, "r") as f:
        lines = f.readlines()
    a, c, b, u_0, v_0  = _read_line_of_floats(lines[0])
    # print(a, c, b)
    k_1, k_2 = _read_line_of_floats(lines[2])

    intrinsic = np.array([
        [a,   c,   u_0],
        [0.0, b,   v_0],
        [0.0, 0.0, 1.0]
    ])

    images = []

    for image_idx in range(5):
        r_lines_idx = range(4 + (image_idx * 5), 7 + (image_idx * 5))
        t_line_idx = 7 + (image_idx * 5)
        r = np.array([_read_line_of_floats(lines[l]) for l in r_lines_idx]).T
        t = np.array([_read_line_of_floats(lines[t_line_idx])]).T

        rt = np.vstack([np.hstack([r, t]), np.array([[0, 0, 0, 1]])])
        images.append({"r": r, "t": t, "extrinsic": rt})
    return {
        "a": a,
        "c": c,
        "b": b,
        "u_0": u_0,
        "v_0": v_0,
        "k_1": k_1,
        "k_2": k_2,

        "intrinsic": intrinsic,

        "images": images,
    }

def undistort_fiducial_points(fiducial_points: Dict, calibration: Dict) -> None:
    for img_points_idx in range(len(fiducial_points["images"])):
        img_points = fiducial_points["images"][img_points_idx]

        undistorted = cv2.undistortPoints(
            img_points["distorted"],
            calibration["intrinsic"],
            # 0 means no tangential distortion
            np.array([calibration["k_1"], calibration["k_2"], 0.0, 0.0]),
            P=calibration["intrinsic"]
        )

        undistorted = undistorted.reshape(-1, 2).T
        img_points["undistorted"] = undistorted


def undistort_image(distorted_img: npt.NDArray, calibration: Dict) -> npt.NDArray:
    return cv2.undistort(
        distorted_img,
        calibration["intrinsic"],
        np.array([calibration["k_1"], calibration["k_2"], 0.0, 0.0]),
    )


def homogeneous_to_euclidian(v):
    return v[:-1, :] / v[-1, :]


def euclidian_to_homogeneous(v):
    _, num_observations = v.shape
    return np.vstack([v, np.ones(num_observations)])


def plot_fiducial_points(
    points: npt.NDArray,
    img: npt.NDArray,
    fn: Path,
    color: str = "red"
) -> None:
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.plot([p[0] for p in points.T], [p[1] for p in points.T], ".", color=color)
    # TODO plot number of fiducial point

    for p_idx, p in enumerate(points.T):
        plt.text(p[0], p[1], p_idx)
        # plt.text(p[0], p[1], p_idx+1)

    plt.savefig(fn, bbox_inches="tight")


def read_img(fn: Path) -> npt.NDArray:
    img = cv2.imread(str(fn))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_image(
    jean_yves_folder: Path,
    img_idx: int,
    calibration: Dict,
    fiducial_points: Dict,
) -> None:
    distorted_img = read_img(jean_yves_folder / f"CalibIm{img_idx}.tif")
    undistorted_img = undistort_image(distorted_img, calibration)

    plot_fiducial_points(
        fiducial_points["images"][img_idx-1]["undistorted"],
        undistorted_img,
        f"img-{img_idx}-undistorted.png",
        "blue",
    )
    plot_fiducial_points(
        fiducial_points["images"][img_idx-1]["distorted"],
        distorted_img,
        f"img-{img_idx}-distorted.png",
        "red",
    )

def get_rows_cols_points(
    fiducial_points: npt.NDArray
) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
    rows = [[] for _ in range(8*2)]
    cols = [[] for _ in range(8*2)]
    for block_start in range(0, 256, 4):
        top_left = block_start
        top_right = block_start + 1
        bottom_right = block_start + 2
        bottom_left = block_start + 3

        block_row_idx = int( (block_start / 4) // 8 )
        block_col_idx = int( (block_start / 4) %  8 )

        # print(block_row_idx, block_col_idx)

        rows[2 * block_row_idx + 0] += [bottom_left, bottom_right]
        rows[2 * block_row_idx + 1] += [top_left, top_right]

        cols[2 * block_col_idx + 0] += [bottom_left, top_left]
        cols[2 * block_col_idx + 1] += [bottom_right, top_right]

    rows_points = []
    for row in rows:
        rows_points.append(np.array([fiducial_points.T[p_idx]for p_idx in row]).T)

    cols_points = []
    for col in cols:
        cols_points.append(np.array([fiducial_points.T[p_idx]for p_idx in col]).T)
    return rows_points, cols_points


def get_singular_vector(M: npt.NDArray) -> npt.NDArray:
    u, s, vh = np.linalg.svd(M)
    return vh.T[:, -1].reshape(-1, 1)


def get_meets_and_joins(camera_points):
    # Num of parallel lines 8 blocks, 2 lines per block, 16 lines total
    rows, cols = get_rows_cols_points(camera_points)

    rows = [row for row in rows]
    cols = [row for row in cols]

    row_joins = []
    for row in rows:
        row_joins.append(get_singular_vector(row.T))

    col_joins = []
    for col in cols:
        col_joins.append(get_singular_vector(col.T))


    row_meet = get_singular_vector(np.hstack(row_joins).T)
    col_meet = get_singular_vector(np.hstack(col_joins).T)
    return (row_meet, col_meet), row_joins + col_joins


def plot_meets_and_joins(
        meets,
        joins,
        img,
        fn,
        x_range: Tuple[int] = (-2000, 1000),
    ):
        plt.figure(figsize=(25, 25))
        plt.imshow(img)

        for line in joins:
            slope = -line[0] / line[1]
            intercept = -line[2] /  line[1]
            # print(slope, intercept)
            line_eq = lambda x: slope * x + intercept
            x_values = np.arange(*x_range, 100)
            y_values = [line_eq(x) for x in x_values]
            plt.plot(x_values, y_values, "-", color="red", linewidth=0.5)

        for point in meets:
            point = homogeneous_to_euclidian(point)
            plt.plot(point[0], point[1], "*", markersize=15)

        plt.axis('equal')
        plt.xlim(x_range)
        plt.ylim(x_range)
        plt.savefig(fn, bbox_inches="tight")


def normalize_to_perspective(
    calibration: Dict,
    points: npt.NDArray,
    apply_skew: bool = True
) -> npt.NDArray:
    a, b = calibration["a"], calibration["b"]
    u_0, v_0 = calibration["u_0"], calibration["v_0"]
    if not apply_skew:
        c_f = 0.0
    else: 
        c_f = calibration["c"] / calibration["b"]

    intrinsic = np.array([
        [1.0, c_f, u_0],
        [0.0, 1.0, v_0],
        [0.0, 0.0, 1.0]
    ])
    # print(intrinsic)
    # print(np.linalg.inv(intrinsic))
    # exit(0)
    return np.linalg.inv(intrinsic) @ points


def compute_focal(vp1: npt.NDArray, vp2: npt.NDArray) -> float:
    u1, v1, w1 = vp1.flatten()
    u2, v2, w2 = vp2.flatten()

    f = np.sqrt((u1*u2 + v1*v2) / -(w1*w2))
    return f


def test_img_3(fiducial_points: Dict, calibration: Dict) -> None:

    img_idx = 2
    world_points = fiducial_points["board"]
    camera_points = fiducial_points["images"][img_idx]["undistorted"]

    data_folder = Path("data/zhang")
    jean_yves_folder = data_folder / "jean-yves"
    distorted_img = read_img(jean_yves_folder / f"CalibIm{img_idx+1}.tif")
    undistorted_img = undistort_image(distorted_img, calibration)
    # plot_fiducial_points(
    #     camera_points,
    #     undistorted_img,
    #     f"lab-3-undistorted.png",
    #     "blue",
    # )


    meets, joins = get_meets_and_joins(euclidian_to_homogeneous(camera_points))
    print(f"Meet of row lines: {homogeneous_to_euclidian(meets[0]).T.tolist()[0]}")
    print(f"Meet of column lines: {homogeneous_to_euclidian(meets[1]).T.tolist()[0]}")
    # plot_meets_and_joins(
    #     meets=meets,
    #     joins=joins,
    #     img=undistorted_img,
    #     fn="meets-and-joins.png",
    #     x_range=(-9000, 1000),
    # )

    # vp1, vp2 = meets

    vp1 = normalize_to_perspective(calibration, meets[0], apply_skew=False)
    vp2 = normalize_to_perspective(calibration, meets[1], apply_skew=False)
    # vp1 = normalize_to_perspective(calibration, meets[0], apply_skew=True)
    # vp2 = normalize_to_perspective(calibration, meets[1], apply_skew=True)
    print(vp1, vp2)

    # print(
    #     homogeneous_to_euclidian(
    #         normalize_to_perspective(calibration, meets[0], apply_skew=True)
    #     )
    # )
    # print(
    #     homogeneous_to_euclidian(
    #         normalize_to_perspective(calibration, meets[1], apply_skew=True)
    #     )
    # )
    # exit(0)

    # norm_camera_points = normalize_to_perspective(
    #     calibration,
    #     euclidian_to_homogeneous(camera_points),
    #     # apply_skew=True,
    #     apply_skew=True,
    # )

    # meets_norm, _ = get_meets_and_joins(norm_camera_points)
    # print(homogeneous_to_euclidian(meets_norm[0]))
    # print(homogeneous_to_euclidian(meets_norm[1]))

    focal = compute_focal(vp1, vp2)
    print(focal)

    # exit(0)
    # print(norm_camera_points.shape)
    # print(camera_points.shape)
    # print(norm_camera_points.mean(1))
    # print(camera_points.mean(1))







def main():
    data_folder = Path("data/zhang")
    jean_yves_folder = data_folder / "jean-yves"
    calibration_fn = data_folder / "completecalibration.txt"

    calibration = read_camera_calibration_matrix(calibration_fn)
    fiducial_points = read_fiducial_points(jean_yves_folder)
    undistort_fiducial_points(fiducial_points, calibration)

    test_img_3(fiducial_points, calibration)


if __name__ == "__main__":
    main()

"""
- [x] Undistort the detected corners using the Brown-Conrady coefficients estimated by Zhang
      How to check if point undistorded properly? - Maybe plot those points on undistorted images

- [x] Using DLT construct the design matrix A for the homogeneous least squares problem and compute the condition number.

- [x] Using SVD, estimate the camera matrix P1.

- [x] Normalize the design matrix and compute the condition number.

- [x] Using SVD, estimate the normalized camera matrix using the result from (4) and then transform it back to the original space to obtain the camera matrix P2.

- [x] Compare the two solutions for the camera matrix that you obtained P1, P2  using the RMS reprojection errror in pixels.

- [x] Compute how well the camera matrices can spatially localize points on the boards by back-projecting the points. Report the RMS errors in board units.

- [x] Recover the intrinsics and extrinsics as specified in "A Flexible New Technique for Camera Calibration."
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

from typing import (Dict,)
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
        plt.text(p[0], p[1], p_idx+1)

    plt.savefig(fn, bbox_inches="tight")


def build_desing_matrix(
    world_points: npt.NDArray,
    camera_points: npt.NDArray
) -> npt.NDArray:

    rows = []
    for wp, cp in zip(world_points.T, camera_points.T):
        wp = wp.reshape(-1, 1)
        cp = cp.reshape(-1, 1)

        wp = euclidian_to_homogeneous(wp)
        cp = euclidian_to_homogeneous(cp)

        zp = np.zeros((3, 1))

        x = cp[0, 0]
        y = cp[1, 0]
        # w = cp[2, 0]

        # rows += [
        #     np.hstack([zp.T, -w*wp.T,  y*wp.T]),
        #     np.hstack([w*wp.T,  zp.T, -x*wp.T]),
        # ]
        rows += [
            np.hstack([zp.T, -wp.T,  y*wp.T]),
            np.hstack([wp.T,  zp.T, -x*wp.T]),
        ]
        # rows += [
        #     np.hstack([-wp.T,  zp.T, x*wp.T]),
        #     np.hstack([zp.T, -wp.T,  y*wp.T]),
        # ]
    A = np.vstack(rows)
    return A


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

def estimate_homography(A: npt.NDArray) -> npt.NDArray:
    u, s, vh = np.linalg.svd(A)
    H = vh.T[:, -1].reshape(-1, 1)
    H = H.reshape(3, 3)
    return H


def get_whitening_transform(points: npt.NDArray) -> npt.NDArray:
    points_cov = np.cov(points)
    d, u = np.linalg.eigh(points_cov)
    d = np.diag(1.0 / np.sqrt(d + 1e-18))
    w = (u @ d) @ u.T
    # w = scipy.linalg.fractional_matrix_power(points_cov, -0.5)
    # w = d @ u.T
    return w


def calc_project_error(p1: npt.NDArray, p2: npt.NDArray) -> float:
    return np.mean(np.sqrt(np.sum((p1 - p2) ** 2, axis=0)))


def apply_H(
    H: npt.NDArray,
    points: npt.NDArray
) -> npt.NDArray:
    projected = H @ euclidian_to_homogeneous(points)
    projected = homogeneous_to_euclidian(projected)
    return projected


def pad_transform(t: npt.NDArray) -> npt.NDArray:
    t = np.pad(t, [(0, 1), (0, 1)], mode='constant')
    t[-1, -1] = 1
    return t


def test_img_3(fiducial_points: Dict, calibration: Dict) -> None:
    # plot_image(jean_yves_folder, 3, calibration, fiducial_points)

    img_idx = 2
    world_points = fiducial_points["board"]
    camera_points = fiducial_points["images"][img_idx]["undistorted"]

    make_plot = True
    if make_plot:
        data_folder = Path("data/zhang")
        jean_yves_folder = data_folder / "jean-yves"
        distorted_img = read_img(jean_yves_folder / f"CalibIm{img_idx+1}.tif")
        undistorted_img = undistort_image(distorted_img, calibration)


    A = build_desing_matrix(
        world_points,
        camera_points,
    )

    A_cond_num = np.linalg.cond(A)
    print(f"Conditional number for A: {A_cond_num}")
    H = estimate_homography(A)

    projected = apply_H(H, world_points)
    H_project_err = calc_project_error(camera_points, projected)
    print(f"Mean projection error in px for H: {H_project_err}")

    back_projected = apply_H(np.linalg.inv(H), camera_points)
    H_back_project_err = calc_project_error(world_points, back_projected)
    print(f"Mean back-project error in inches for H: {H_back_project_err}")
    if make_plot:
        plot_fiducial_points(
            projected,
            undistorted_img,
            f"img-{img_idx}-undistorted-with-projected-points.png",
            "blue",
        )



    w_world = get_whitening_transform(world_points)
    w_camera = get_whitening_transform(camera_points)
    # print(np.cov(w_world @ world_points))
    # print(np.cov(w_camera @ camera_points))
    # print(w_world.shape)

    A_norm = build_desing_matrix(
        w_world @ world_points,
        w_camera @ camera_points,
    )

    A_norm_cond_num = np.linalg.cond(A_norm)
    print(f"Conditional number for normalized A: {A_norm_cond_num}")
    H_tilda_norm = estimate_homography(A_norm)

    # w_world_homogeneous = np.pad(w_world, [(0, 1), (0, 1)], mode='constant')
    # w_world_homogeneous[-1, -1] = 1
    # w_camera_homogeneous = np.pad(w_camera, [(0, 1), (0, 1)], mode='constant')
    # w_camera_homogeneous[-1, -1] = 1
    w_world_homogeneous = pad_transform(w_world)
    w_camera_homogeneous = pad_transform(w_camera)

    H_norm = np.linalg.inv(w_camera_homogeneous) @ H_tilda_norm @ w_world_homogeneous

    projected_norm = apply_H(H_norm, world_points)
    H_norm_project_err = calc_project_error(camera_points, projected_norm)
    print(f"Mean project error for H after normalization: {H_norm_project_err}")

    back_projected_norm = apply_H(np.linalg.inv(H_norm), camera_points)
    H_norm_back_project_err = calc_project_error(world_points, back_projected_norm)
    print(f"Mean back-project error in inches for H after normalization: {H_norm_back_project_err}")

    if make_plot:
        plot_fiducial_points(
            projected_norm,
            undistorted_img,
            f"img-{img_idx}-undistorted-with-projected-points-norm.png",
            "blue",
        )


def calculate_homography_per_image(
    world_points: npt.NDArray,
    camera_points: npt.NDArray,
    normalize: bool = True,
) -> npt.NDArray:
    if not normalize:
        A = build_desing_matrix(
            world_points,
            camera_points,
        )
        H = estimate_homography(A)

    else:
        w_world = get_whitening_transform(world_points)
        w_camera = get_whitening_transform(camera_points)

        A = build_desing_matrix(
            w_world @ world_points,
            w_camera @ camera_points,
        )

        H_tilda = estimate_homography(A)

        w_world_homogeneous = pad_transform(w_world)
        w_camera_homogeneous = pad_transform(w_camera)

        H = np.linalg.inv(w_camera_homogeneous) @ H_tilda @ w_world_homogeneous

    return H


def recover_intrinsics(fiducial_points: Dict) -> npt.NDArray:
    H_per_image = [
        calculate_homography_per_image(
            fiducial_points["board"],
            points["undistorted"],
            normalize=True,
        )
        for points in fiducial_points["images"]
    ]


    def build_v_vector(H: npt.NDArray, i: int, j: int):
        return np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[1, i] * H[1, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j],
        ]).reshape(-1, 1)


    rows = []
    for H in H_per_image:
        v_11 = build_v_vector(H, 0, 0)
        v_12 = build_v_vector(H, 0, 1)
        v_22 = build_v_vector(H, 1, 1)

        rows += [
            v_12.T,
            v_11.T - v_22.T,
        ]
    V = np.vstack(rows)

    u, s, vh = np.linalg.svd(V)
    b = vh.T[:, -1].reshape(-1)

    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])



    # A = np.linalg.cholesky(B)
    # K = np.linalg.inv(A).T

    def getK(B):
        """
        sorry, get this code from https://github.com/sakshikakde/zhang-camera-calibration/blob/main/Code/Utils/MathUtils.py,
        chilesky decomposition was not working correctly, will fix it later :)
        """
        v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2])/(B[0,0] * B[1,1] - B[0,1]**2)
        lamb = B[2,2] - (B[0,2]**2 + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2]))/B[0,0]
        alpha = np.sqrt(lamb/B[0,0])
        beta = np.sqrt(lamb * (B[0,0]/(B[0,0] * B[1,1] - B[0,1]**2)))
        gamma = -(B[0,1] * alpha**2 * beta) / lamb 
        u0 = (gamma * v0 / beta) - (B[0,2] * alpha**2 / lamb)

        K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        return K

    K = getK(B)
    return K



def main():
    data_folder = Path("data/zhang")
    jean_yves_folder = data_folder / "jean-yves"
    calibration_fn = data_folder / "completecalibration.txt"

    calibration = read_camera_calibration_matrix(calibration_fn)
    fiducial_points = read_fiducial_points(jean_yves_folder)
    undistort_fiducial_points(fiducial_points, calibration)

    test_img_3(fiducial_points, calibration)

    intrinsics = recover_intrinsics(fiducial_points)
    print("Recovered intrinsics")
    print(intrinsics)

    print("Zhang intrinsics")
    print(calibration["intrinsic"])


if __name__ == "__main__":
    main()

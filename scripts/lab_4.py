"""
Topic: Stereo Reconstruction and MLE
- [ ] Implement the normalized 8-point algorithm

- [ ] Recover E from F by using the gold-standard calibration from Zhang.

- [ ] Recover R,t from E by resolving the 4-fold triangulation ambiguity.

- [ ] Triangulate the correspondences of the tin can.

- [ ] Plot the reconstruction and camera poses with fields of view.

- [ ] Minimally parameterize the relative pose (R,t)

- [ ] Use black box non-linear least squares to minimize the re-projection error

- [ ] Plot the MLE reconstruction and camera poses with fields of view.

- [ ] Compare the re-projection error before and after non-linear least squares.

To match practice with theory, compare the results to the unnormalized estimate of F
so that you might realize the contribution of Hartley's paper
"""
import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import cv2
import scipy
import math

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import (Dict, Tuple, List)
from pathlib import Path


def homogeneous_to_euclidian(v):
    return v[:-1, :] / v[-1, :]


def euclidian_to_homogeneous(v):
    _, num_observations = v.shape
    return np.vstack([v, np.ones(num_observations)])


def read_camera_calibration_matrix(fn: Path) -> Dict:
    _read_line_of_floats = lambda l: list(map(float, l.split(" ")))

    with open(fn, "r") as f:
        lines = f.readlines()
    a, c, b, u_0, v_0  = _read_line_of_floats(lines[0])
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


def read_img(fn: Path) -> npt.NDArray:
    img = cv2.imread(str(fn))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_features(fn: Path) -> npt.NDArray:
    features = []
    with open(fn, "r") as f:
        for line in f:
            features.append([float(v) for v in line.split(" ")])
    features = np.array(features).T
    return features


def to_pixel_coords(
    features: npt.NDArray,
    imgs: npt.NDArray,
) -> npt.NDArray:
    updated_features = []
    for points, img in zip(features, imgs):
        img_h, img_w, _ = img.shape
        points *= np.array([img_w, img_h]).reshape(-1, 1)
        updated_features.append(points)
    return updated_features


def plot_teatin_with_features(img, features, img_idx) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.plot(features[0], features[1], "*", color="red")
    for p_idx, p in enumerate(features.T):
        plt.text(p[0], p[1], p_idx)
    plt.savefig(f"teatin-features-img-{img_idx+1}.png", bbox_inches="tight")


def build_coef_matrix_A(
    features1: npt.NDArray,
    features2: npt.NDArray
) -> npt.NDArray:

    rows = []
    for p1, p2 in zip(features1.T, features2.T):
        p1 = p1.reshape(-1, 1)
        p2 = p2.reshape(-1, 1)

        x, y, w = p2.flatten()

        # print(p1.T)
        # print(p2.T)
        # print(np.hstack([x * p1.T, y * p1.T, w * p1.T]))
        # exit(0)
        rows.append(np.hstack([x * p1.T, y * p1.T, w * p1.T]))

    A = np.vstack(rows)
    return A


def get_singular_vector(M: npt.NDArray) -> npt.NDArray:
    u, s, vh = np.linalg.svd(M)
    return vh.T[:, -1].reshape(-1, 1)


def get_whitening_transform(points: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    mean = points.mean(axis=1).reshape(-1, 1)
    points_cov = np.cov(points - mean)
    d, u = np.linalg.eigh(points_cov)
    d = np.diag(1.0 / np.sqrt(d + 1e-18))
    w = (u @ d) @ u.T
    return w, mean


def pad_whitening(t: npt.NDArray, m: npt.NDArray) -> npt.NDArray:
    m = np.pad(-m, [(0, 1), (2, 0)])
    np.fill_diagonal(m, 1)
    t = np.pad(t, [(0, 1), (0, 1)], mode='constant')
    t[-1, -1] = 1
    return t @ m

    # w_world, world_mean = get_whitening_transform(world_points)
    # w_camera, camera_mean = get_whitening_transform(camera_points)
    # w_world_homogeneous = pad_whitening(w_world, world_mean)
    # w_camera_homogeneous = pad_whitening(w_camera, camera_mean)



def calc_fundamental_matrix(
    features1: npt.NDArray,
    features2: npt.NDArray
) -> npt.NDArray:
    features1 = euclidian_to_homogeneous(features1)
    features2 = euclidian_to_homogeneous(features2)
    # TODO normalized points
    A = build_coef_matrix_A(features1, features2)
    fa = get_singular_vector(A)
    Fa = fa.reshape(3, 3)
    u, s, vh = np.linalg.svd(Fa)
    s[-1] = 0.0
    F = u @ np.diag(s) @ vh
    return F

    # print(u)
    # print(s)
    # print(vh)
    # print()
    # print(Fa)
    # print(u @ np.diag(s) @ vh)
    # print(A.shape)


def calc_essentail_matrix(
    fundamental: npt.NDArray,
    intrinsic: npt.NDArray
)-> npt.NDArray:
    return np.linalg.inv(intrinsic) @ fundamental @ intrinsic


def main():
    data_folder = Path("data/zhang")
    jean_yves_folder = data_folder / "jean-yves"

    calibration_fn = data_folder / "completecalibration.txt"
    calibration = read_camera_calibration_matrix(calibration_fn)

    tin_img_filenames = [
        data_folder / "teatin1.png",
        data_folder / "teatin2.png",
    ]
    tin_feature_filenames = [
        data_folder / "image1text.txt",
        data_folder / "image2text.txt",
    ]
    tin_imgs = [read_img(fn) for fn in tin_img_filenames]
    tin_features = [read_features(fn) for fn in tin_feature_filenames]
    tin_features = to_pixel_coords(tin_features, tin_imgs)


    F = calc_fundamental_matrix(*tin_features)
    # img_idx = 1
    # img = tin_imgs[img_idx]
    # features = tin_features[img_idx]
    # plot_teatin_with_features(img, features, img_idx)


if __name__ == "__main__":
    main()

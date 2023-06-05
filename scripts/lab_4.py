"""
Topic: Stereo Reconstruction and MLE
- [x] Implement the normalized 8-point algorithm

- [x] Recover E from F by using the gold-standard calibration from Zhang.

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

# import matplotlib
# matplotlib.use('Agg')

import cv2
import scipy
import math

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import (Dict, Tuple, List)
from pathlib import Path
from itertools import product


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


def get_mean_and_whitening(points: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    mean = points.mean(axis=1).reshape(-1, 1)
    points_cov = np.cov(points - mean)
    d, u = np.linalg.eigh(points_cov)
    d = np.diag(1.0 / np.sqrt(d + 1e-18))
    w = (u @ d) @ u.T
    return mean, w


def compose_normalization_transform(
    m: npt.NDArray,
    t: npt.NDArray,
)-> npt.NDArray:

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
    m, w = get_mean_and_whitening(np.hstack([features1, features2]))
    norm_t = compose_normalization_transform(m, w)

    features1 = euclidian_to_homogeneous(features1)
    features2 = euclidian_to_homogeneous(features2)

    # print(features1)
    # features1 = norm_t @ features1
    # features1 = np.linalg.inv(norm_t) @ features1
    # print(features1)
    # exit(0)

    features1 = norm_t @ features1
    features2 = norm_t @ features2

    A = build_coef_matrix_A(features1, features2)
    fa = get_singular_vector(A)
    Fa = fa.reshape(3, 3)
    u, s, vh = np.linalg.svd(Fa)
    s[-1] = 0.0
    Fn = u @ np.diag(s) @ vh

    F = norm_t.T @ Fn @ norm_t
    # F = np.linalg.inv(norm_t).T @ Fn @ np.linalg.inv(norm_t)
    return F

    # # Without normalization
    # features1 = euclidian_to_homogeneous(features1)
    # features2 = euclidian_to_homogeneous(features2)
    # A = build_coef_matrix_A(features1, features2)
    # fa = get_singular_vector(A)
    # Fa = fa.reshape(3, 3)
    # u, s, vh = np.linalg.svd(Fa)
    # s[-1] = 0.0
    # F = u @ np.diag(s) @ vh
    # return F

def calc_essentail_matrix(
    fundamental: npt.NDArray,
    intrinsic: npt.NDArray
)-> npt.NDArray:
    return intrinsic.T @ fundamental @ intrinsic
    # return np.linalg.inv(intrinsic) @ fundamental @ intrinsic


def to_vector(v): return v.reshape(-1, 1)

def triangulate_point(
    feature_1,
    feature_2,
    camera_matrix_1,
    camera_matrix_2,
):

    # TODO add whitening
    x1, y1, _ = feature_1.flatten()
    x2, y2, _ = feature_2.flatten()

    p1 = camera_matrix_1[1]

    rows = [
        x1 * to_vector(camera_matrix_1[2]).T - to_vector(camera_matrix_1[0]).T,
        y1 * to_vector(camera_matrix_1[2]).T - to_vector(camera_matrix_1[1]).T,
        x2 * to_vector(camera_matrix_2[2]).T - to_vector(camera_matrix_2[0]).T,
        y2 * to_vector(camera_matrix_2[2]).T - to_vector(camera_matrix_2[1]).T,
    ]


    A = np.vstack(rows)
    # print(A)
    x = get_singular_vector(A)
    x = homogeneous_to_euclidian(x)
    return x

# def triangulate_points(
#     features_1,
#     features_2,
#     camera_matrix_1,
#     camera_matrix_2,
# )


def find_valid_R_and_t(
    R_candidates: List[npt.NDArray],
    t_candidates: List[npt.NDArray],
    sample_feature: Tuple[npt.NDArray, npt.NDArray],
) -> Tuple[npt.NDArray, npt.NDArray]:
    for R, t in list(product(R_candidates, t_candidates))[:]:
    # for R, t in product(R_candidates, t_candidates):
        # print(R)
        # print(t)
        # rt = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])
        # test_vector = np.array([0, 0, 1, 1]).reshape(-1, 1)

        camera_matrix_1 = np.hstack([np.identity(3), np.zeros((3, 1))])
        camera_matrix_2 = np.hstack([R, t])
        # print(homogeneous_to_euclidian(rt @ test_vector))
        # print(camera_matrix_1)

        x = triangulate_point(*sample_feature, camera_matrix_1, camera_matrix_2)
        # TODO fix
        # print(x)
        # print()

    return R, t


def recover_rotation_and_translation(
    essential: npt.NDArray,
    sample_feature: Tuple[npt.NDArray, npt.NDArray],
) -> Tuple[npt.NDArray, npt.NDArray]:
    u, s, vh = np.linalg.svd(essential)
    u3 = u[:, -1].reshape(-1, 1)
    w = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    # z = np.array([
    #     [ 0.0, 1.0, 0.0],
    #     [-1.0, 0.0, 0.0],
    #     [ 0.0, 0.0, 0.0],
    # ])
    R_candidates = [ u @ w @ vh, u @ w.T @ vh, ]
    t_candidates = [ u3, -u3, ]

    R, t = find_valid_R_and_t(R_candidates, t_candidates, sample_feature)
    return R, t


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
    # # TODO how to check if F is valid?
    # pair_idx = 0
    # p1 = euclidian_to_homogeneous(tin_features[0][:, pair_idx].reshape(-1, 1))
    # p2 = euclidian_to_homogeneous(tin_features[1][:, pair_idx].reshape(-1, 1))
    # l2 = F @ p1
    # print(l2.T @ p2)
    # l1 = (p2.T @ F).T
    # print(l1.T @ p1)
    # exit(0)
    E = calc_essentail_matrix(F, calibration["intrinsic"])
    # print(E)
    # u, s, vh = np.linalg.svd(E)
    # print(s)
    # print(u)
    # print(vh)

    sample_feature = [
        np.linalg.inv(calibration["intrinsic"]) \
        @ euclidian_to_homogeneous(to_vector(tin_features[0][:, 0])),

        np.linalg.inv(calibration["intrinsic"]) \
        @ euclidian_to_homogeneous(to_vector(tin_features[1][:, 0])),
    ]

    R1, t1 = np.identity(3), np.zeros((3, 1))
    camera_matrix_1 = np.hstack([R1, t1])

    R2, t2 = recover_rotation_and_translation(E, sample_feature)
    camera_matrix_2 = np.hstack([R2, t2])

    features1, features2 = tin_features
    features1 = euclidian_to_homogeneous(features1)
    features2 = euclidian_to_homogeneous(features2)
    K = calibration["intrinsic"]
    features1 = np.linalg.inv(K) @ features1
    features2 = np.linalg.inv(K) @ features2

    triangulated = []
    for f1, f2 in zip(features1.T, features2.T):
        x = triangulate_point(f1, f2, camera_matrix_1, camera_matrix_2)
        triangulated.append(x.flatten())

    print(triangulated)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.plot(
        [p[0] for p in triangulated],
        [p[1] for p in triangulated],
        [p[2] for p in triangulated],
        "*",
        color="green",
        label="z axis",
    )

    for p_idx, p in enumerate(triangulated):
        ax.text(*p, f"{p_idx}")
    plt.show()



    # img_idx = 1
    # img = tin_imgs[img_idx]
    # features = tin_features[img_idx]
    # plot_teatin_with_features(img, features, img_idx)


if __name__ == "__main__":
    main()

"""
Topic: Stereo Reconstruction and MLE, part II

- [x] Minimally parameterize the relative pose (R,t)

- [x] Use black box non-linear least squares to minimize the re-projection error

- [x] Plot the MLE reconstruction and camera poses with fields of view.

- [x] Compare the re-projection error before and after non-linear least squares.
"""

import sys, os
sys.path.append(os.getcwd())

import cv2 # opencv-python-headless==4.7.0.72
import scipy # scipy==1.10.1
import math

import numpy as np # numpy==1.24.3
import numpy.typing as npt
import matplotlib.pyplot as plt # to draw 2d
import plotly.graph_objects as go # to draw 3d

from typing import (Dict, Tuple, List)
from pathlib import Path
from itertools import product
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from more_itertools import pairwise


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


def build_coef_matrix_A(
    features1: npt.NDArray,
    features2: npt.NDArray
) -> npt.NDArray:

    rows = []
    for p1, p2 in zip(features1.T, features2.T):
        p1 = p1.reshape(-1, 1)
        p2 = p2.reshape(-1, 1)

        x, y, w = p2.flatten()
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


def calc_fundamental_matrix(
    features1: npt.NDArray,
    features2: npt.NDArray
) -> npt.NDArray:
    m, w = get_mean_and_whitening(features1)
    norm_t1 = compose_normalization_transform(m, w)

    m, w = get_mean_and_whitening(features2)
    norm_t2 = compose_normalization_transform(m, w)

    features1 = euclidian_to_homogeneous(features1)
    features2 = euclidian_to_homogeneous(features2)

    # print(features1)
    # features1 = norm_t @ features1
    # features1 = np.linalg.inv(norm_t) @ features1
    # print(features1)
    # exit(0)

    features1 = norm_t1 @ features1
    features2 = norm_t2 @ features2

    A = build_coef_matrix_A(features1, features2)
    fa = get_singular_vector(A)
    Fa = fa.reshape(3, 3)
    u, s, vh = np.linalg.svd(Fa)
    s[-1] = 0.0

    Fn = u @ np.diag(s) @ vh

    F = norm_t2.T @ Fn @ norm_t1
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


def to_vector(v): return v.reshape(-1, 1)


cond_number_a = []
def triangulate_point(
    f1,
    f2,
    rt1,
    rt2,
):
    x1, y1, _ = f1.flatten()
    x2, y2, _ = f2.flatten()

    rows = [
        x1 * to_vector(rt1[2]).T - to_vector(rt1[0]).T,
        y1 * to_vector(rt1[2]).T - to_vector(rt1[1]).T,
        x2 * to_vector(rt2[2]).T - to_vector(rt2[0]).T,
        y2 * to_vector(rt2[2]).T - to_vector(rt2[1]).T,
    ]


    A = np.vstack(rows)

    # Nomalize A by cols (suggested by James)
    norm_a_by_cols = np.linalg.inv(np.diag(np.linalg.norm(A, ord=np.inf, axis=0)))
    A = A @ norm_a_by_cols
    x = get_singular_vector(A)
    x = norm_a_by_cols @ x

    # # Nomalize A by rows (inspired by stackoverflow answer)
    # norm_a_by_rows = np.linalg.inv(np.diag(np.linalg.norm(A, ord=np.inf, axis=1)))
    # A = norm_a_by_rows @ A
    # x = get_singular_vector(A)

    # # Without normalization
    # A = np.vstack(rows)
    # x = get_singular_vector(A)

    # print(f"Cond number of A (triangulation): {np.linalg.cond(A)}")
    cond_number_a.append(np.linalg.cond(A))
    return x


def triangulate_set_of_points(
    features1,
    features2,
    rt1,
    rt2,
    K,
):
    features1 = euclidian_to_homogeneous(features1)
    features2 = euclidian_to_homogeneous(features2)
    features1 = np.linalg.inv(K) @ features1
    features2 = np.linalg.inv(K) @ features2

    m, w = get_mean_and_whitening(homogeneous_to_euclidian(features1))
    norm_t1 = compose_normalization_transform(m, w)
    m, w = get_mean_and_whitening(homogeneous_to_euclidian(features2))
    norm_t2 = compose_normalization_transform(m, w)

    features1 = norm_t1 @ features1
    features2 = norm_t2 @ features2
    rt1 = norm_t1 @ rt1
    rt2 = norm_t2 @ rt2

    triangulated = []
    for f1, f2 in zip(features1.T, features2.T):
        x = homogeneous_to_euclidian(
            triangulate_point(f1, f2, rt1, rt2)
        )
        triangulated.append(x.flatten())
    return triangulated


def find_valid_R_and_t(
    R_candidates: List[npt.NDArray],
    t_candidates: List[npt.NDArray],
    sample_feature: Tuple[npt.NDArray, npt.NDArray],
) -> Tuple[npt.NDArray, npt.NDArray]:
    # return R_candidates[0], t_candidates[0]
    # return R_candidates[1], t_candidates[1]
    # return R_candidates[0], t_candidates[1]
    # return R_candidates[1], t_candidates[0]

    for R, t in product(R_candidates, t_candidates):
        rt = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])
        test_vector = np.array([0, 0, 1, 1]).reshape(-1, 1)

        rt1 = np.vstack([
            np.hstack([np.identity(3), np.zeros((3, 1))]),
            np.array([[0, 0, 0, 1]])
        ])
        rt2 = np.vstack([
            np.hstack([R, t]),
            np.array([[0, 0, 0, 1]])
        ])


        cam1_x = triangulate_point(*sample_feature, rt1, rt2)
        cam2_x = rt2 @ cam1_x

        cam1_x = homogeneous_to_euclidian(cam1_x)
        cam2_x = homogeneous_to_euclidian(cam2_x)

        # Check if point located in front of camera for both cam1 and cam2
        if (cam1_x.flatten()[-1] >= 0.0) and (cam2_x.flatten()[-1] >= 0.0):
            return R, t

    raise Exception("No valid R and t")


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
    R_candidates = [ u @ w @ vh, u @ w.T @ vh, ]
    t_candidates = [ u3, -u3, ]

    R, t = find_valid_R_and_t(R_candidates, t_candidates, sample_feature)
    return R, t


def is_rotation(R):
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float32))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one


# def undistort_image(distorted_img: npt.NDArray, calibration: Dict) -> npt.NDArray:
#     return cv2.undistort(
#         distorted_img,
#         calibration["intrinsic"],
#         np.array([calibration["k_1"], calibration["k_2"], 0.0, 0.0]),
#     )


# def undistort_features(features: npt.NDArray, calibration: Dict) -> npt.NDArray:

#     undistorted = cv2.undistortPoints(
#         features,
#         calibration["intrinsic"],
#         # 0 means no tangential distortion
#         np.array([calibration["k_1"], calibration["k_2"], 0.0, 0.0]),
#         P=calibration["intrinsic"]
#     )

#     undistorted = undistorted.reshape(-1, 2).T
#     return undistorted


# Bundle adjustment stuff
def reprojection_error(
    features1,
    features2,
    cam1,
    cam2,
    triangulated,
) -> float:
    """
    calculate mean reprojection error in px
    """
    # TODO is it valid to use such error to measure effect of normalization of A (triangulation)
    # becase there are still cam1 and cam2 that also have impact on error score

    errors = []
    for f1, t in zip(features1.T, triangulated.T):
        f1 = to_vector(f1)
        t = to_vector(t)
        projected_t = homogeneous_to_euclidian(cam1 @ euclidian_to_homogeneous(t))
        err = np.linalg.norm(projected_t - f1, ord=2)
        errors.append(err)

    for f2, t in zip(features2.T, triangulated.T):
        f2 = to_vector(f2)
        t = to_vector(t)
        projected_t = homogeneous_to_euclidian(cam2 @ euclidian_to_homogeneous(t))
        err = np.linalg.norm(projected_t - f2, ord=2)
        errors.append(err)

    error = np.mean(errors)
    return error


def skew(v: npt.NDArray) -> npt.NDArray:
    v = v.flatten()
    assert len(v) == 3, f"only len 3 vector allowed"
    return np.array([
        [  0.0,  -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],   v[0],   0.0],
    ])


def rotataion_matrix_to_axis_angle(R: npt.NDArray) -> npt.NDArray:
    # print(R)
    v = get_singular_vector(R - np.identity(3))
    rrt = (R - R.T)
    vh = np.array([rrt[2, 1], rrt[0, 2], rrt[1, 0]]).reshape(-1, 1)
    theta = np.arctan2(
        np.dot(v.flatten(), vh.flatten()) / 2.0,
        (np.trace(R) - 1) / 2.0,
    )
    r = v * theta
    # r, _ = cv2.Rodrigues(R)
    return r


def rotataion_axis_angle_to_matrix(axis_angle: npt.NDArray) -> npt.NDArray:
    theta = np.linalg.norm(axis_angle)
    v = axis_angle / theta

    R =  np.identity(3) \
      + np.sin(theta) * skew(v) \
      + (1 - np.cos(theta)) * np.linalg.matrix_power(skew(v), 2)
    # R, _ = cv2.Rodrigues(axis_angle)
    # print(R)
    return R


def translation_spherical_to_cartesian(t):
    theta, phi = t.flatten().tolist()

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return to_vector(np.array([x, y, z]))


def translation_cartesian_to_spherical(t):
    x, y, z = t.flatten().tolist()
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    if x <= 0:
        phi = np.arctan2(y, x)

    elif x > 0 and y >= 0:
        phi = np.arctan2(y, x) + np.pi

    elif x > 0 and y < 0:
        phi = np.arctan2(y, x) - np.pi
    return to_vector(np.array([theta, phi]))


def bundle_adjustment(
    R1, t1,
    R2, t2,
    K,
    features1,
    features2,
    triangulated,
) -> None:
    xdata =  [ (1, p_idx) for p_idx in range(len(features1.T))] \
          +  [ (2, p_idx) for p_idx in range(len(features2.T))]

    ydata = np.vstack([
        features1.T,
        features2.T,
    ])
    ydata = ydata.flatten()

    t2_len = np.linalg.norm(t2)
    t2_unit = t2 / t2_len

    p0 = [
        *rotataion_matrix_to_axis_angle(R2).flatten().tolist(),
        *translation_cartesian_to_spherical(t2_unit).flatten().tolist(),
    ]
    p0 = p0 + [0.0 for _ in range(len(triangulated.T) * 3)]

    print(f"Num of parameters to optimize: {len(p0)}")

    def to_optimize(indep, r1, r2, r3, t1, t2, *point_adjustment):
        out = []
        for i in indep:
            cam_idx, point_idx = i
            cam_idx = int(cam_idx)
            point_idx = int(point_idx)

            if cam_idx == 1:
                R, t = np.identity(3), np.zeros((3, 1))
                # R, t = R1, t1

            elif cam_idx == 2:
                r = to_vector(np.array([r1, r2, r3]))
                R = rotataion_axis_angle_to_matrix(r)
                t_sp = to_vector(np.array([t1, t2]))
                t = translation_spherical_to_cartesian(t_sp) * t2_len

            rt = np.hstack([R, t])
            P = K @ rt

            X_adj = point_adjustment[point_idx * 3: (point_idx+1) * 3]
            X_adj = to_vector(np.array(X_adj))
            X_init = to_vector(triangulated.T[point_idx])

            X = X_init + X_adj
            X = euclidian_to_homogeneous(X)

            x = P @ X
            x = homogeneous_to_euclidian(x)

            out.append(x)

        out = np.hstack(out).T
        out = out.flatten()
        return out

    popt, pcov = curve_fit(to_optimize, xdata, ydata, p0)
    r1_adj, r2_adj, r3_adj, t1_adj, t2_adj, *point_adj = popt
    R_adj = rotataion_axis_angle_to_matrix(to_vector(np.array([r1_adj, r2_adj, r3_adj])))

    t_sp = to_vector(np.array([t1_adj, t2_adj]))
    t_adj = translation_spherical_to_cartesian(t_sp) * t2_len

    triangulated_adj = []
    for point_idx, X_init in enumerate(triangulated.T):
        X_init = to_vector(X_init)
        X_adj = point_adj[point_idx * 3: (point_idx+1) * 3]
        X_adj = to_vector(np.array(X_adj))
        X = X_init + X_adj
        triangulated_adj.append(X.flatten())

    return R_adj, t_adj, triangulated_adj


# Visualization stuff
def plot_teatin_with_features(img, features, img_idx) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.plot(features[0], features[1], "*", color="red")
    for p_idx, p in enumerate(features.T):
        plt.text(p[0], p[1], p_idx)
    plt.savefig(f"teatin-features-img-{img_idx+1}-undistorted.png", bbox_inches="tight")


def get_colors(triangulated, tin_imgs, features1, features2):
    triangulated_colors = [
        np.mean([
            tin_imgs[0][int(f1[0]), int(f1[1])],
            tin_imgs[1][int(f2[0]), int(f2[1])],
        ], axis=0).astype(int)
        for f1, f2 in zip(features1.T, features2.T)
    ]
    triangulated_colors = [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in triangulated_colors]
    return triangulated_colors


def draw_fov(camera_origin, fov_points, fov_color):
    camera_origin = camera_origin.flatten()
    fov_points = [p.flatten() for p in fov_points]

    fov = []
    for fov_p in fov_points:
        fov.append(go.Scatter3d(
            x=[camera_origin[0], fov_p[0]],
            y=[camera_origin[1], fov_p[1]],
            z=[camera_origin[2], fov_p[2]],
            mode="lines",
            line=dict(
                color=fov_color,
            ),
        ))

    for (fov_p1, fov_p2) in pairwise(fov_points + [fov_points[0]]):
        fov.append(go.Scatter3d(
            x=[fov_p1[0], fov_p2[0]],
            y=[fov_p1[1], fov_p2[1]],
            z=[fov_p1[2], fov_p2[2]],
            mode="lines",
            line=dict(
                color=fov_color,
            ),
        ))
    return fov


def draw_basis(origin, x, y, z, name, fov_points=None, fov_color=None):

    origin = origin.flatten()
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    basis = [
        go.Scatter3d(
            x=[origin[0], x[0]],
            y=[origin[1], x[1]],
            z=[origin[2], x[2]],
            mode="lines",
            line=dict(
                color="red",
            )
        ),
        go.Scatter3d(
            x=[origin[0], y[0]],
            y=[origin[1], y[1]],
            z=[origin[2], y[2]],
            mode="lines",
            line=dict(
                color="blue",
            )
        ),
        go.Scatter3d(
            x=[origin[0], z[0]],
            y=[origin[1], z[1]],
            z=[origin[2], z[2]],
            mode="lines",
            line=dict(
                color="green",
            )
        ),
    ]
    origin_text_shift = np.array([-0.05, -0.05, -0.05])
    basis += [
        go.Scatter3d(
            x=[(origin+origin_text_shift)[0]],
            y=[(origin+origin_text_shift)[1]],
            z=[(origin+origin_text_shift)[2]],
            text=name,
            mode="text",
        ),
        go.Scatter3d(
            x=[x[0]],
            y=[x[1]],
            z=[x[2]],
            text="x",
            mode="text",
        ),
        go.Scatter3d(
            x=[y[0]],
            y=[y[1]],
            z=[y[2]],
            text="y",
            mode="text",
        ),
        go.Scatter3d(
            x=[z[0]],
            y=[z[1]],
            z=[z[2]],
            text="z",
            mode="text",
        ),
    ]

    if fov_points is not None:
        basis += draw_fov(origin, fov_points, fov_color)
    return basis


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


def draw_cameras(R2, t2, calibration):
    cam1_origin = np.array([[0, 0, 0]]).T * 0.2
    cam1_z_axis = np.array([[0, 0, 1]]).T * 0.2
    cam1_y_axis = np.array([[0, 1, 0]]).T * 0.2
    cam1_x_axis = np.array([[1, 0, 0]]).T * 0.2

    cam2_extrinsics = np.vstack([np.hstack([R2, t2]), np.array([[0, 0, 0, 1]])])

    cam2_origin = np.linalg.inv(cam2_extrinsics) @ euclidian_to_homogeneous(cam1_origin)
    cam2_z_axis = np.linalg.inv(cam2_extrinsics) @ euclidian_to_homogeneous(cam1_z_axis)
    cam2_y_axis = np.linalg.inv(cam2_extrinsics) @ euclidian_to_homogeneous(cam1_y_axis)
    cam2_x_axis = np.linalg.inv(cam2_extrinsics) @ euclidian_to_homogeneous(cam1_x_axis)


    cam2_origin = homogeneous_to_euclidian(cam2_origin)
    cam2_z_axis = homogeneous_to_euclidian(cam2_z_axis)
    cam2_y_axis = homogeneous_to_euclidian(cam2_y_axis)
    cam2_x_axis = homogeneous_to_euclidian(cam2_x_axis)

    fov_color = "lightblue"
    fov_scale = 15
    fov_points = calculate_fov_points(
        calibration,
        cam1_z_axis,
        fov_scale
    )

    fov1_points = fov_points
    fov2_points = [np.linalg.inv(cam2_extrinsics) @ euclidian_to_homogeneous(fp) 
                   for fp in fov_points]
    fov2_points = [homogeneous_to_euclidian(fp) for fp in fov2_points]

    basis1 = draw_basis(
        cam1_origin,
        cam1_x_axis,
        cam1_y_axis,
        cam1_z_axis,
        f"Cam 1",
        fov_points=fov1_points,
        fov_color=fov_color,
    )

    basis2 = draw_basis(
        cam2_origin,
        cam2_x_axis,
        cam2_y_axis,
        cam2_z_axis,
        f"Cam 2",
        fov_points=fov2_points,
        fov_color=fov_color,
    )
    return basis1 + basis2


def draw_triangulated_points(triangulated, triangulated_colors):
    return [go.Scatter3d(
        x=[p[0] for p in triangulated],
        y=[p[1] for p in triangulated],
        z=[p[2] for p in triangulated],
        text=[str(i) for i in range(len(triangulated))],

        marker=dict(
            size=5,
            symbol="square",
            color=triangulated_colors,
        ),
        mode="markers+text"
    )]


def draw_3d_plot(triangulated, triangulated_colors, R2, t2, calibration):
    fig = go.Figure(
        data=draw_triangulated_points(triangulated, triangulated_colors) \
           + draw_cameras(R2, t2, calibration)
    )

    fig.update_layout(
        showlegend=False,
        hovermode=False,
        autosize=True,
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                eye=dict(x=0.2, y=1, z=-2)
            ),
            aspectratio = dict(x=1, y=1, z=1),
            aspectmode = 'data',
            yaxis_autorange="reversed",
            xaxis_autorange="reversed",
            xaxis=dict(
                showgrid=True,
                backgroundcolor="white",
                gridcolor="lightgray", 
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=True,
                backgroundcolor="white",
                gridcolor="lightgray", 
                showticklabels=False,
            ),
            zaxis=dict(
                showgrid=True,
                backgroundcolor="white",
                gridcolor="lightgray", 
                showticklabels=False,
            ),
        ),
    )
    return fig


def main():
    # Read all required data
    data_folder = Path("data/zhang")
    jean_yves_folder = data_folder / "jean-yves"

    calibration_fn = data_folder / "completecalibration.txt"
    calibration = read_camera_calibration_matrix(calibration_fn)
    K = calibration["intrinsic"]

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

    # Calculate Fundamental and Essential matrix
    F = calc_fundamental_matrix(*tin_features)
    E = calc_essentail_matrix(F, K)

    # Recover R and t from E
    sample_feature = [
        np.linalg.inv(K) @ euclidian_to_homogeneous(to_vector(tin_features[0][:, 0])),
        np.linalg.inv(K) @ euclidian_to_homogeneous(to_vector(tin_features[1][:, 0])),
    ]

    R1, t1 = np.identity(3), np.zeros((3, 1))
    rt1 = np.hstack([R1, t1])

    R2, t2 = recover_rotation_and_translation(E, sample_feature)
    rt2 = np.hstack([R2, t2])

    # Triangulate all points on a tin
    features1, features2 = tin_features
    triangulated = triangulate_set_of_points(
        features1,
        features2,
        rt1,
        rt2,
        K,
    )

    # Calculate re-projection error
    reproject_error_score = reprojection_error(
        features1, features2,
        K @ rt1, K @ rt2,
        np.vstack(triangulated).T,
    )
    print(f"Re-projection error: {reproject_error_score:.6f}")
    print(f"Mean cond number of A during trinagulation: {np.mean(cond_number_a)}")

    # Bundle adjustment
    do_bundle_adjustment = True
    if do_bundle_adjustment:
        R2_adj, t2_adj, triangulated_adj = bundle_adjustment(
            R1, t1,
            R2, t2,
            K,
            features1,
            features2,
            np.vstack(triangulated).T,
        )

        rt2_adj = np.hstack([R2_adj, t2_adj])
        reproject_error_score_after_adj = reprojection_error(
            features1, features2,
            K @ rt1, K @ rt2_adj,
            np.vstack(triangulated_adj).T,
        )
        print(f"Re-projection error after bundle adj: {reproject_error_score_after_adj:.6f}")

        R2 = R2_adj
        t2 = t2_adj
        triangulated = triangulated_adj

    # Draw plots
    do_drawing = True
    if do_drawing:
        plot_teatin_with_features(tin_imgs[0], tin_features[0], 0)
        plot_teatin_with_features(tin_imgs[1], tin_features[1], 1)
        triangulated_colors = get_colors(
            triangulated,
            tin_imgs,
            features1,
            features2
        )
        fig = draw_3d_plot(
            triangulated,
            triangulated_colors,
            R2,
            t2,
            calibration,
        )
        fig.write_html("tea-tin-stereo-reconstruction.html")
        # fig.show()


if __name__ == "__main__":
    main()

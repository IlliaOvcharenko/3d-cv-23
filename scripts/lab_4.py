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


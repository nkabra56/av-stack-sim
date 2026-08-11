# KITTI data attribution

`excerpt_poses.txt` is a 300-frame excerpt (frames 840-1139) of ground-truth poses from
**sequence 09** of the **KITTI Vision Benchmark Suite** odometry benchmark, used here to
validate `core/estimation/ekf.py` against a real driven trajectory (see
`core/validation/`, DESIGN.md's "Validation against real data" section).

Format is unmodified KITTI: each line is 12 numbers, a row-major 3x4 `[R|t]` pose matrix in
the left-camera frame at that timestep, relative to frame 0 of the *original, full* sequence
09 (not re-zeroed to this excerpt) — so absolute positions here are wherever frames 840-1139
happen to fall in that original trajectory, which is fine for this project's purposes (only
relative motion and position differences are used).

Source: [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_odometry.php),
downloaded from the public ground-truth-poses archive.

Please cite the following papers if you use this data:

```
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0).
This excerpt is redistributed here for non-commercial, educational/portfolio use only.

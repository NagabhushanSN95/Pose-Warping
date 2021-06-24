# Pose-Warping for View Synthesis / DIBR
Code to warp a frame from a source view to the target view, given the depth map of source view, camera poses of both views and camera intrinsic matrix.

### Features:
1. The warping is implemented in python using numpy/pytorch package and has been vectorized. 
2. It uses inverse bilinear interpolation (which can be considered as a trivial form of splatting). 
3. Pytorch implementation is differentiable.
4. Also contains splatting/interpolation code given flow and/or depth.
5. Supports warping of masked frame/flow i.e. if frame/flow is unknown at certain pixels.

### Other implementations:
1. [Reference View Synthesizer (RVS)](https://gitlab.com/mpeg-i-visual/rvs)
2. [Open MVS](https://github.com/cdcseacave/openMVS)
3. A splatting based differential warping has been implemented by [PyTorch 3D](https://github.com/facebookresearch/pytorch3d) which has been used in [SynSin](https://github.com/facebookresearch/synsin) 
4. [Softmax Splatting](https://github.com/sniklaus/softmax-splatting)


### Citation
If you find this work useful, please cite it as 

```bibtex
@misc{somraj2020posewarping,
    title = {Pose-Warping for View Synthesis / {DIBR}},
    author = {Somraj, Nagabhushan},
    year = {2020},
    url = {https://github.com/NagabhushanSN95/Pose-Warping}
}
```

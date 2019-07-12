# GeoNet
[GeoNet: Geometric Neural Network for Joint Depth and Surface Normal Estimation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_GeoNet_Geometric_Neural_CVPR_2018_paper.pdf)

GeoNet++: Interative Geometric Neural network with Edge-aware Refinement Joint Depth and Surface Normal Estimation. (In preparation)

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.2) + Scipy + Numpy + Scipy + OpenCV.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.


### Inference
1. Download pretrained model data, initialization model, and trained model from "https://drive.google.com/open?id=1o2t8735acVf2cLSCS6URkNViOB7mdb-Q"

2. tar xvzf GeoNet.tar.gz

3. Merge files into the Repo according to the file name.

4. Run 'code.py'

5. Evaluation: cd eval & run 'test_depth.m' for depth evaluation and run 'test_norm.py' for normal evaluation.

### Training
Training code already included. Detailed data preparation & running procedure will be updated soon.
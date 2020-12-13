# GeoNet
[GeoNet: Geometric Neural Network for Joint Depth and Surface Normal Estimation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_GeoNet_Geometric_Neural_CVPR_2018_paper.pdf)

[GeoNet++: Iterative Geometric Neural network with Edge-aware Refinement Joint Depth and Surface Normal Estimation](https://ieeexplore.ieee.org/document/9184024)

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
Prepared training data download: 

https://hkuhk-my.sharepoint.com/personal/xjqi_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxjqi%5Fhku%5Fhk%2FDocuments%2FGeoNet%2Ddepth%2Dnormal%2Ddata&originalPath=aHR0cHM6Ly9oa3Voay1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC94anFpX2hrdV9oay9FazBWbS0tNW9pMUdzc2lvTEU1TGpPMEJ5TFRLcFdBRzAwellZVUNlaXlkUjdnP3J0aW1lPVVSZ3RVVEtmMkVn)

Run 'code.py' in training mode.


### Citation

If you find the code or trained models useful, please consider citing:

```
@inproceedings{qi2018geonet,
  title={Geonet: Geometric neural network for joint depth and surface normal estimation},
  author={Qi, Xiaojuan and Liao, Renjie and Liu, Zhengzhe and Urtasun, Raquel and Jia, Jiaya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={283--291},
  year={2018}
}
@article{qi2020geonet++,
  title={GeoNet++: Iterative Geometric Neural Network with Edge-Aware Refinement for Joint Depth and Surface Normal Estimation},
  author={Qi, Xiaojuan and Liu, Zhengzhe and Liao, Renjie and Torr, Philip HS and Urtasun, Raquel and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}

```

### Question

You are welcome to send pull requests or give some advices. Contact information: `xjqi at eee.hku.hk`.
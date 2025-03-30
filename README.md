# RING++: Roto-Translation-Invariant Gram for Global Localization on a Sparse Scan Map (IEEE T-RO 2023)
Official implementation of RING and RING++:
* [One RING to Rule Them All: Radon Sinogram for Place Recognition, Orientation and Translation Estimation](https://ieeexplore.ieee.org/document/9981308) (IEEE IROS 2022).
* [RING++: Roto-Translation-Invariant Gram for Global Localization on a Sparse Scan Map](https://ieeexplore.ieee.org/document/10224330) (IEEE T-RO 2023).

## Abstract
Global localization plays a critical role in many robot applications. LiDAR-based global localization draws the community’s focus with its robustness against illumination and seasonal changes. To further improve the localization under large viewpoint differences, we propose RING++ that has roto-translation invariant representation for place recognition and global convergence for both rotation and translation estimation. With the theoretical guarantee, RING++ is able to address the large viewpoint difference using a lightweight map with sparse scans. In addition, we derive sufficient conditions of feature extractors for the representation preserving the roto-translation invariance, making RING++ a framework applicable to generic multichannel features. To the best of our knowledge, this is the first learning-free framework to address all the subtasks of global localization in the sparse scan map. Validations on real-world datasets show that our approach demonstrates better performance than state-of-the-art learning-free methods and competitive performance with learning-based methods. Finally, we integrate RING++ into a multirobot/session simultaneous localization and mapping system, performing its effectiveness in collaborative applications.

![framework](imgs/framework.png)
## revised code description
- We modified evaluate.py and plot_curve.py to evaluate the place recognition ability of Ring and to better visualize the results.
- We changed KITTIDataset.py, and the source code seems to have some problems with selecting keyframes.
- NOTE：When conducting the place recognition evaluation, we set the map_sampling_distance and query_sampling_distance to the same value to determine the value of exclude_recent_nodes.In addition, this prevents the history frame timestamp from being larger than the query frame timestamp.
## Usage
- For code usage and library installation, you can refer:https://github.com/lus6-Jenny/RING
### Plot
To plot the precision-recall curve and trajectory path, run:
```bash
python evaluation/plot_curve.py 
```
NOTE: You may need to change the path of the results in the script.

## Citation
If you find this work useful, please cite:
```bibtex
@article{xu2023ring++,
  title={RING++: Roto-Translation-Invariant Gram for Global Localization on a Sparse Scan Map},
  author={Xu, Xuecheng and Lu, Sha and Wu, Jun and Lu, Haojian and Zhu, Qiuguo and Liao, Yiyi and Xiong, Rong and Wang, Yue},
  journal={IEEE Transactions on Robotics},
  year={2023},
  publisher={IEEE}
}
```
```bibtex
@inproceedings{lu2023deepring,
  title={DeepRING: Learning Roto-translation Invariant Representation for LiDAR based Place Recognition},
  author={Lu, Sha and Xu, Xuecheng and Tang, Li and Xiong, Rong and Wang, Yue},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1904--1911},
  year={2023},
  organization={IEEE}
}
```
```bibtex
@inproceedings{lu2022one,
  title={One ring to rule them all: Radon sinogram for place recognition, orientation and translation estimation},
  author={Lu, Sha and Xu, Xuecheng and Yin, Huan and Chen, Zexi and Xiong, Rong and Wang, Yue},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2778--2785},
  year={2022},
  organization={IEEE}
}
```

## Contact
If you have any questions, please contact
```
Sha Lu: lusha@zju.edu.cn
```

## License
The code is released under the [MIT License](https://opensource.org/license/mit/).

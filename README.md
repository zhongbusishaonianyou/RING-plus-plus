# RING++: Roto-Translation-Invariant Gram for Global Localization on a Sparse Scan Map (IEEE T-RO 2023)
Official implementation of RING and RING++:
* [One RING to Rule Them All: Radon Sinogram for Place Recognition, Orientation and Translation Estimation](https://ieeexplore.ieee.org/document/9981308) (IEEE IROS 2022).
* [RING++: Roto-Translation-Invariant Gram for Global Localization on a Sparse Scan Map](https://ieeexplore.ieee.org/document/10224330) (IEEE T-RO 2023).

# Abstract
Global localization plays a critical role in many robot applications. LiDAR-based global localization draws the community’s focus with its robustness against illumination and seasonal changes. To further improve the localization under large viewpoint differences, we propose RING++ that has roto-translation invariant representation for place recognition and global convergence for both rotation and translation estimation. With the theoretical guarantee, RING++ is able to address the large viewpoint difference using a lightweight map with sparse scans. In addition, we derive sufficient conditions of feature extractors for the representation preserving the roto-translation invariance, making RING++ a framework applicable to generic multichannel features. To the best of our knowledge, this is the first learning-free framework to address all the subtasks of global localization in the sparse scan map. Validations on real-world datasets show that our approach demonstrates better performance than state-of-the-art learning-free methods and competitive performance with learning-based methods. Finally, we integrate RING++ into a multirobot/session simultaneous localization and mapping system, performing its effectiveness in collaborative applications.

![framework](imgs/framework.png)
# Revised code description
- We modified evaluate.py and plot_curve.py to evaluate the place recognition ability of Ring and to better visualize the results.
- We changed KITTIDataset.py, and the source code seems to have some problems with selecting keyframes.
- NOTE：When conducting the place recognition evaluation, we set the map_sampling_distance and query_sampling_distance to the same value to determine the value of exclude_recent_nodes.In addition, this prevents the history frame timestamp from being larger than the query frame timestamp.
# Usage
- For code usage and library installation, you can refer:https://github.com/lus6-Jenny/RING
# Plot
To plot the precision-recall curve and trajectory path, run:
```bash
python evaluation/plot_curve.py 
```
NOTE: You may need to change the path of the results in the script.
# PR result
 - ## KITTI results (kitti 02)
|                                query_sampling_distance=2m    |          query_sampling_distance=5m                          |    
| ------------------------------------------------------------ | ------------------------------------------------------------ | 
|![Figure_12](https://github.com/user-attachments/assets/e5b19d9b-34d5-4722-8394-ab05cf253888)| ![Figure_6](https://github.com/user-attachments/assets/cab5e76f-a3ca-4ebb-89fe-eb8babcdb661) |
|![Figure_7](https://github.com/user-attachments/assets/a098c71f-7569-4b3c-87c9-70bfd74240dc) | ![Figure_10](https://github.com/user-attachments/assets/bfe0dcdb-b4f0-45be-bce6-f71afe3e9652)|
|![Figure_9](https://github.com/user-attachments/assets/13b74873-8dff-46f2-b7e5-ff6a548ad406)|![Figure_11](https://github.com/user-attachments/assets/1b68a19a-98e5-4cbf-8a9e-728fb159351c)|

 - ## NCLT results(2012-08-20)
|                                   query_sampling_distance=2m |    query_sampling_distance=5m                                |            
| ------------------------------------------------------------ | ------------------------------------------------------------ | 
|![Figure_4](https://github.com/user-attachments/assets/644aa40c-20b7-4008-9d06-c288447cbb29)| ![Figure_5](https://github.com/user-attachments/assets/b5f1c4e5-8000-47ab-9bd1-13bf2c865a23)|
|![Figure_0](https://github.com/user-attachments/assets/f6b76a66-a6ad-472f-af2a-c537b3fe49d1)|![Figure_1](https://github.com/user-attachments/assets/8ea8a52a-b75a-4ac6-bdd5-f304c5b185c5) |
|![Figure_2](https://github.com/user-attachments/assets/9e79f3d9-a4cf-478c-a05e-b859aed2ae9a)| ![Figure_3](https://github.com/user-attachments/assets/3f3cd4c3-06d9-4695-9e66-10011fe6e02a)|
 - ## MulRan results(DCC02)
|                                   query_sampling_distance=2m |    query_sampling_distance=5m                                |            
| ------------------------------------------------------------ | ------------------------------------------------------------ | 
|![Figure_24](https://github.com/user-attachments/assets/283b0555-e4e0-483e-8e88-24e00e54fde2)|![Figure_13](https://github.com/user-attachments/assets/f04ee11c-2ea0-4f57-8c7a-1f25488a5fe7)|
|![Figure_14](https://github.com/user-attachments/assets/8df08481-bf18-441a-b97d-b3c32e8a7fb8)|![Figure_15](https://github.com/user-attachments/assets/a27302d8-df21-4c1f-94d8-8d2ddb2eee1c)|
|![Figure_16](https://github.com/user-attachments/assets/95f8facc-2515-434b-a570-5a5edb356600)| ![Figure_17](https://github.com/user-attachments/assets/81531d08-f3ee-45a1-bf80-1ffdf230c408)|
# result analysis
In fact, what we have shown above is the result of a single channel utilizing a RING that occupies information. According to the experimental results, RING performs better with a larger sampling interval.The main advantage of RING is that it can handle revisits with large lateral translation and has a relatively accurate estimation of the three-degree-of-freedom pose.The advantage of RING is not obvious when the revisit threshold is set small or the sampling distance is selected small.
# Citation
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

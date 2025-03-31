import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from sklearn.neighbors import KDTree
from utils.core import *
import utils.config as cfg
from utils.tools import imshow
from utils.poses import relative_pose, xyz_ypr2m, m2xyz_ypr, angle_clip, cal_pose_error
from utils.point_clouds import o3d_icp, fast_gicp, draw_pc, draw_pc_pair, draw_registration_result

from datasets.NCLTDataset import NCLTPointCloudLoader
from datasets.KITTIDataset import KITTIPointCloudLoader
from datasets.MulRanDataset import MulRanPointCloudLoader
from datasets.OxfordRadarDataset import OxfordRadarPointCloudLoader

from evaluation.plot_pose_errors import plot_cdf
from evaluation.plot_curve import calculate_dist
from evaluation.generate_evaluation_sets import EvaluationSet, EvaluationTuple


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)


def generate_representations(dataset: str, eval_subset: List[EvaluationTuple], bev_type: str = "occ"):
    # Generate BEV & RING representations for the evaluation set
    # Each BEV / RING representation is a numpy array of shape (C, H, W)
    # H: height, W: width, C: number of channels
    # C = 1 for occupancy BEV
    # C = 6 for feature BEV
    
    pcs = []
    pc_bevs = []
    pc_RINGs = []
    pc_TIRINGs = []
    if dataset == 'nclt':
        pc_loader = NCLTPointCloudLoader()
    elif dataset == 'kitti':
        pc_loader = KITTIPointCloudLoader()
    elif dataset == 'mulran':
        pc_loader = MulRanPointCloudLoader()
    elif dataset == 'oxford_radar':
        pc_loader = OxfordRadarPointCloudLoader()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    for ndx, e in tqdm(enumerate(eval_subset)):
        pc = pc_loader.read_pc(e.rel_scan_filepath)
        pc = pc[:,:3]
        pc_norm = pc_loader.normalize_pc(pc)
        
        # Generate BEV features
        if device == torch.device("cpu"):
            pc_bev = generate_bev(pc) # BEV encoded by occupancy information
            pc_RING, pc_TIRING = generate_RING_cpu(pc_bev)
        else:      
            if bev_type == "occ":
                pc_bev = generate_bev_occ(pc_norm) # BEV encoded by occupancy information
            elif bev_type == "feat":
                pc_bev = generate_bev_feats(pc_norm) # BEV encoded by point features
            else:
                raise ValueError('bev_type is not "occ" or "feat"')
            pc_RING, pc_TIRING = generate_RING(pc_bev)
            
        pcs.append(pc)
        pc_bevs.append(pc_bev)
        pc_RINGs.append(pc_RING)
        pc_TIRINGs.append(pc_TIRING)
    
    return pcs, pc_bevs, pc_RINGs, pc_TIRINGs


def evaluate(dataset, eval_set_filepath: str, revisit_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0], num_k: int = 25, bev_type: str = "occ"):
    
    assert os.path.exists(eval_set_filepath), f'Cannot access evaluation set pickle: {eval_set_filepath}'
    eval_set = EvaluationSet()
    eval_set.load(eval_set_filepath)
    query_set = eval_set.query_set
    query_xys = eval_set.get_query_positions()
    query_poses = eval_set.get_query_poses()
    num_queries = len(query_xys)
    query_pcs, query_bevs, query_RINGs, query_TIRINGs = generate_representations(dataset, query_set, bev_type)
    

    pair_dists =np.zeros((num_queries-cfg.exclude_recent_nodes, 3))
    rot_errors = np.zeros((len(revisit_thresholds), num_queries)) # rotation errors
    trans_errors = np.zeros((len(revisit_thresholds), num_queries))  # translation errors
    icp_rot_errors = np.zeros((len(revisit_thresholds), num_queries))  # rotation errors after ICP
    icp_trans_errors = np.zeros((len(revisit_thresholds), num_queries))  # translation errors after ICP    
    
    for query_ndx in tqdm(range(cfg.exclude_recent_nodes,num_queries)):
        query_xy = query_xys[query_ndx]
        query_pose = query_poses[query_ndx]
        query_pc = query_pcs[query_ndx]
        query_bev = query_bevs[query_ndx]

        C, H, W = query_bevs[0].shape

        query_TIRING = query_TIRINGs[query_ndx]
        query_TIRING_repeated = query_TIRING.repeat(cfg.batch_size, 1, 1, 1)
        
        # ------------ Place Recognition ------------
        if (query_ndx -cfg.exclude_recent_nodes+1) % cfg.batch_size == 0:
            batch_num = (query_ndx -cfg.exclude_recent_nodes+1)// cfg.batch_size
        else:
            batch_num = (query_ndx -cfg.exclude_recent_nodes+1) // cfg.batch_size + 1     
        print(f"query_index/total_queries: {query_ndx+1}/{num_queries}")
        for i in range(batch_num):
            
            if i == batch_num - 1:
                history_TIRING = [query_TIRINGs[k] for k in range(i*cfg.batch_size, query_ndx -cfg.exclude_recent_nodes+1)]
                history_TIRING = torch.stack(history_TIRING, dim=0).reshape((-1, C, H, W))
                query_TIRING_repeated = query_TIRING.repeat(history_TIRING.shape[0], 1, 1, 1)
            else:
                history_TIRING = [query_TIRINGs[k] for k in range(i*cfg.batch_size, (i+1)*cfg.batch_size)] 
                history_TIRING = torch.stack(history_TIRING, dim=0).reshape((-1, C, H, W))
           
            batch_dists, batch_angles = batch_circorr(query_TIRING_repeated, history_TIRING)
            
            if i == 0:
                dists = batch_dists
                angles = batch_angles
            else:
                dists = np.concatenate((dists, batch_dists), axis=-1)
                angles = np.concatenate((angles, batch_angles), axis=-1)
        
        dists = dists.squeeze()
        angles = angles.squeeze() 

        if  dists.ndim == 0: 
           dists=np.array([dists]) # 标量 
        
        idxs_sorted = np.argsort(dists)
        idx_top1 = idxs_sorted[0]
        pair_dists[query_ndx - cfg.exclude_recent_nodes] = [query_ndx,idx_top1,dists[idx_top1]] 
        
         # ------------ Pose Estimation ------------
             # Perform pose estimation for the top 1 match within the revisit threshold
        for j, revisit_threshold in enumerate(revisit_thresholds): 
            distance = calculate_dist(query_xy, query_xys[idx_top1]) 
            if  distance< revisit_threshold:
                history_pose = query_poses[idx_top1]
                history_pc = query_pcs[idx_top1]
                history_bev = query_bevs[idx_top1]
                angle_matched = angles[idx_top1]
                rel_pose = relative_pose(query_pose, history_pose)
                gt_x, gt_y, gt_z, gt_yaw, gt_pitch, gt_roll = m2xyz_ypr(rel_pose)
                #print(f"-------- Query {query_ndx+1}th frame matched with history {idx_top1+1}th frame--------")
                #print("Ground truth translation: x: {}, y: {}, rotation: {}".format(gt_x, gt_y, gt_yaw))

                ang_res = 2 * np.pi / cfg.num_ring # angular resolution
                # angle between the two matched RINGs in grids
                angle_matched_extra = angle_matched - cfg.num_ring // 2
                # convert the matched angle from grids to radians
                angle_matched_rad = angle_matched * ang_res 
                angle_matched_extra_rad = angle_matched_extra * ang_res        

                bev_rotated = rotate_bev(query_bev, angle_matched_rad)
                bev_rotated_extra = rotate_bev(query_bev, angle_matched_extra_rad)

                # solve the translation between the two matched bevs
                x, y, error = solve_translation(bev_rotated, history_bev)
                x_extra, y_extra, error_extra = solve_translation(bev_rotated_extra, history_bev)

                if error < error_extra:
                    pred_x = x / cfg.num_sector * (cfg.point_cloud["x_bound"][1] - cfg.point_cloud["x_bound"][0])  # in meters
                    pred_y = y / cfg.num_ring * (cfg.point_cloud["y_bound"][1] - cfg.point_cloud["y_bound"][0])   # in meters
                    pred_yaw = angle_matched_rad  # in radians
                else:
                    pred_x = x_extra / cfg.num_sector * (cfg.point_cloud["x_bound"][1] - cfg.point_cloud["x_bound"][0])  # in meters
                    pred_y = y_extra / cfg.num_ring * (cfg.point_cloud["y_bound"][1] - cfg.point_cloud["y_bound"][0])  # in meters 
                    pred_yaw = angle_matched_extra_rad  # in radians
                
               # print("RING Estimated translation: x: {}, y: {}, rotation: {}".format(pred_x, pred_y, pred_yaw))

                yaw_err = np.abs(angle_clip(gt_yaw - pred_yaw)) * 180 / np.pi # in degrees
                x_err = np.abs(gt_x - pred_x) # in meters
                y_err = np.abs(gt_y - pred_y) # in meters
                trans_err = np.sqrt(x_err**2 + y_err**2) # in meters
                rot_errors[j, query_ndx] = yaw_err
                trans_errors[j, query_ndx] = trans_err

                # ------------ Pose Refinement ------------
                init_pose = xyz_ypr2m(pred_x, pred_y, 0, pred_yaw, 0, 0)
                times = time.time()
                icp_fitness_score, loop_transform = fast_gicp(query_pc,history_pc, max_correspondence_distance=cfg.icp_max_distance, init_pose=init_pose)
                # icp_fitness_score, loop_transform, _ = o3d_icp(query_pc, map_pc, transform=init_pose, point2plane=True, inlier_dist_threshold=cfg.icp_max_distance)
                #  timee = time.time()    
                # print("ICP processed time:", timee - times, 's')
                x, y, z, yaw, pitch, roll = m2xyz_ypr(loop_transform)
               # print("fast_gicp Refined translation: x: {}, y: {}, rotation: {}".format(x, y, yaw))

                icp_rte, icp_rre = cal_pose_error(loop_transform, rel_pose)
                icp_rot_errors[j, query_ndx] = icp_rre
                icp_trans_errors[j, query_ndx] = icp_rte
    
    eval_setting = eval_set_filepath.split('/')[-1].split('.pickle')[0]
    folder = f"./results/{dataset}/{eval_setting}/revisit"
    for j, revisit_threshold in enumerate(revisit_thresholds):
        folder = f"{folder}_{revisit_threshold}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # used for drawing pr curve
    np.savetxt(f'{folder}/pair_dists.txt', pair_dists, fmt="%d %d %.6f")
    np.savetxt(f'{folder}/poses.txt',query_xys, fmt="%.6f")
    
    
     # ------------ Pose Error ------------ 
    mean_rot_error = np.mean(rot_errors[j])
    mean_trans_error = np.mean(trans_errors[j])
    mean_icp_rot_error = np.mean(icp_rot_errors[j])
    mean_icp_trans_error = np.mean(icp_trans_errors[j])    
       
    print(f"Mean rotation error at {revisit_threshold} m: {mean_rot_error}")    
    print(f"Mean translation error at {revisit_threshold} m: {mean_trans_error}")
    print(f"Mean icp rotation error at {revisit_threshold} m: {mean_icp_rot_error}")    
    print(f"Mean icp translation error at {revisit_threshold} m: {mean_icp_trans_error}")    
    

    save_path = f'{folder}/rot_error_cdf_{bev_type}_{revisit_threshold}m.pdf'
    plot_cdf(rot_errors[j], save_path=save_path, xlabel='Rotation Error (degrees)', ylabel='CDF', title='Rotation error CDF')
    np.savetxt(f'{folder}/rot_errors_{bev_type}.txt', rot_errors)
    np.savetxt(f'{folder}/trans_errors_{bev_type}.txt', trans_errors)
    np.savetxt(f'{folder}/icp_rot_errors_{bev_type}.txt', icp_rot_errors)
    np.savetxt(f'{folder}/icp_trans_errors_{bev_type}.txt', icp_trans_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nclt", help="dataset type (nclt / mulran / kitti / oxford_radar)")
    parser.add_argument('--eval_set_filepath', type=str, required=True, help='File path of the evaluation pickle')
    parser.add_argument('--revisit_thresholds', type=float, nargs='+', default=[5.0, 10.0, 15.0, 20.0], help='Revisit thresholds in meters')
    parser.add_argument('--num_k', type=int, default=25, help='Number of nearest neighbors for recall@k')
    parser.add_argument('--bev_type', type=str, default="occ", help='BEV type (occ / feat)')
    args = parser.parse_args()

    print(f'Dataset:\033[34m {args.dataset}\033[0m')
    print(f'Evaluation set path: \033[34m{args.eval_set_filepath}\033[0m')
    print(f'Revisit thresholds [m]:\033[34m{args.revisit_thresholds}\033[0m')
    print(f'BEV type:\033[34m{args.bev_type}\033[0m')

    evaluate(args.dataset, args.eval_set_filepath, args.revisit_thresholds, num_k=args.num_k, bev_type=args.bev_type)

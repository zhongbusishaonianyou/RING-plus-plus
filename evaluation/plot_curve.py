import argparse
import matplotlib
import utils.config as cfg
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf', size=20)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.weight': "bold"})
matplotlib.rcParams.update({'axes.labelweight': "bold"})
matplotlib.rcParams.update({'axes.titleweight': "bold"})
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import KDTree


def calculate_dist(pose1, pose2):
    dist = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist

def calculate_loop_truth(poses,revisit_threshold):
    
    GT_Loop=np.zeros((poses.shape[0]-cfg.exclude_recent_nodes), dtype=int)
    for index in range(cfg.exclude_recent_nodes,poses.shape[0]): 
        query_pos=poses[index]
        for jj in range(index-cfg.exclude_recent_nodes+1):
           distance = calculate_dist(query_pos, poses[jj])
           if (distance < revisit_threshold):
                GT_Loop[index-cfg.exclude_recent_nodes]=1
                break
           else:
                GT_Loop[index-cfg.exclude_recent_nodes]=0
    num_positive = np.sum(GT_Loop == 1)
    print(f'the number of revisit events is {num_positive} with {revisit_threshold}m') 
    return GT_Loop, num_positive

def visualize_trajectory(poses,GT_Loop, loop_results,R_P100,idx,revisit_threshold):
    plt.figure(idx)
    plt.title(f'revisit_threshold={revisit_threshold}m',fontproperties=font)
    plt.xlabel('x', fontproperties=font, fontstyle='normal')
    plt.ylabel('y',fontproperties=font, fontstyle='normal')
    plt.tick_params(labelsize=18)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.plot(poses[:,1],poses[:,0],  "k", linewidth=1.5)

    for i in range(len(loop_results)):
       if GT_Loop[i]:
         index =int( loop_results[i,0])
         plt.scatter(poses[index,1], poses[index,0], c="g")
       if loop_results[i,2] <= R_P100 and GT_Loop[i] == 1:
         index = int(loop_results[i][0])
         plt.scatter(poses[index,1], poses[index,0], c="r")
       if loop_results[i,2] <= R_P100 and GT_Loop[i] == 0:
         index = int(loop_results[i][0])
         plt.scatter(poses[index,1], poses[index,0], c="b")

def compute_PR_pairs(pair_dists, query_positions, num_positive,thresholds, revisit_threshold: float = 10.0):

    REC=[]
    PRE=[]
    F1_score=[]                 
    for j,thre in enumerate(thresholds):
        predict_posive=0 
        TP=0        
        for i in range(pair_dists.shape[0]):
            similarity_dis = pair_dists[i,2]
            query_position = query_positions[i+cfg.exclude_recent_nodes]
            real_dist = calculate_dist(query_position, query_positions[ int(pair_dists[i,1])]) 
            if similarity_dis <= thre:
                predict_posive=predict_posive+1
                if (real_dist < revisit_threshold):
                    TP = TP + 1
        precision = TP * 1.0 /predict_posive
        recall = TP * 1.0 / num_positive   
        F1 = 2 * (precision * recall) / (precision + recall)
        REC.append(recall)
        PRE.append(precision)
        F1_score.append(F1)
    
    for i in range(len(thresholds)-1):
      if PRE[i]==1.0 and PRE[i+1]!=1.0:
        viz_loop_thres = thresholds[i]
        break 
      
    return PRE, REC, F1_score,viz_loop_thres

def compute_AP(precisions, recalls):
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1])*precisions[i]
    return ap


def compute_AUC(precisions, recalls):
    auc = 0.0
    for i in range(len(precisions) - 1):
        auc += (recalls[i+1] - recalls[i])*(precisions[i+1] + precisions[i]) / 2.0
    return auc


if __name__ == '__main__':
    # settings
    bev_type = 'occ' # 'occ' or 'feat'
    method = 'RING' if bev_type == 'occ' else 'RING++'
    results_path = f'./results/kitti/test_02_02_5.0_5.0_5.0/revisit_5.0_10.0_15.0_20.0'
    revisit_thresholds = results_path.split('/')[-1].split('_')[1:]
    revisit_thresholds=np.array(revisit_thresholds, dtype=float)
    # load results
    Loop_results = np.loadtxt(f'{results_path}/pair_dists.txt')
    poses = np.loadtxt(f'{results_path}/poses.txt')
    
    min_thres=np.min(Loop_results[:,2])
    max_thres=np.max(Loop_results[:,2]) 
    #print(f"Min distance: {min_thres}, Max distance: {max_thres}")
    thresholds = np.linspace(min_thres, max_thres, 100)  
    precisions_all = np.zeros((len(revisit_thresholds), len(thresholds)))
    recalls_all = np.zeros((len(revisit_thresholds), len(thresholds)))
    for idx, revisit_threshold in enumerate(revisit_thresholds):   
        GT_Loop,num_positive=calculate_loop_truth(poses, revisit_threshold)
        precisions, recalls, F1_scores,viz_thres = compute_PR_pairs(Loop_results, poses,num_positive,thresholds, revisit_threshold)
        visualize_trajectory(poses,GT_Loop,Loop_results,viz_thres,idx,revisit_threshold)
        precisions_all[idx] = precisions
        recalls_all[idx] = recalls
   
    plt.figure(4)   
    markers = ['o', 's', 'x', 'v', 'd', 'p', 'h']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for idx,revisit_threshold in enumerate(revisit_thresholds):
       plt.plot(recalls_all[idx,:], precisions_all[idx], marker = markers[idx], color = colors[idx], linewidth = 2.0, label = f'{method}({revisit_threshold}m)')
    
    plt.legend(loc = 'lower left')
    new_ticks=np.linspace(0,1,11)
    plt.xticks(new_ticks,fontproperties=font)
    plt.yticks(new_ticks,fontproperties=font)
    plt.title('Precision-Recall Curve',fontproperties=font)
    # plot minor gridlines
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--', alpha=0.2)
    plt.grid(which='major', linestyle='-', alpha=0.2)
    # convert major tick lines from out direction to in direction
    plt.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    # remove minor tick lines and labels
    plt.tick_params(axis='both', which='minor', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.xlabel('Recall',color='b', fontproperties=font, fontstyle='normal')
    plt.ylabel('Precision',color='b', fontproperties=font, fontstyle='normal')
    plt.savefig(f'{results_path}/precision_recall_curve_{bev_type}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()        
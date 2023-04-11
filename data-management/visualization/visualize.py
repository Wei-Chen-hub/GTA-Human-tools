import torch
import trimesh
import smplx
import os

import open3d as o3d
import numpy as np

from gen_pointcloud import process_sequence_ptc


def create_kp_diff_lines(param_path, frame_idx):
    param = dict(np.load(param_path, allow_pickle=True))
    keys = param.keys()

    assert {'pd_kp', 'gt_kp'}.issubset(keys), 'Predicted and ground truth key points needed'

    pd_kps = param['pd_kp'][frame_idx]
    gt_kps = param['gt_kp'][frame_idx]

    objs_pd, objs_gt, lines = [], [], []

    for pd, gt in zip(pd_kps, gt_kps):
        if 0 in gt: continue
        pd_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(pd)
        pd_sphere.paint_uniform_color([0., 0., 1.])

        gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(gt)
        gt_sphere.paint_uniform_color([0., 1., 0.])
        objs_pd.append(pd_sphere)
        objs_gt.append(gt_sphere)

        qs = np.stack([pd, gt])  # pd_incomp,
        line_set_idx = [[0, 1]]  # , [1, 2]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(qs),
            lines=o3d.utility.Vector2iVector(line_set_idx))
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 1]])
        lines.append(line_set)
    return objs_pd + objs_gt + lines  # pd_incomp_sphere


def revert_smplx_hands_pca(param_dict, num_pca_comps):
    # gta-human++ 24
    hl_pca = param_dict['left_hand_pose']
    hr_pca = param_dict['right_hand_pose']

    smplx_model = dict(np.load(r'\\wsl$\Ubuntu-20.04\home\weichen\zoehuman\mmhuman3d'
                               r'\data\body_models\smplx\SMPLX_NEUTRAL.npz', allow_pickle=True))

    hl = smplx_model['hands_componentsl'] # 45, 45
    hr = smplx_model['hands_componentsr'] # 45, 45

    hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1)
    hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

    hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32)
    hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

    param_dict['left_hand_pose'] = hl_reverted
    param_dict['right_hand_pose'] = hr_reverted

    return param_dict


def create_smpl_o3d_obj(param_path, frame_idx):
    # read a npz file, return smpl or smplx model

    param = dict(np.load(param_path, allow_pickle=True))
    keys = param.keys()

    assert {'global_orient', 'transl', 'body_pose', 'betas'}.issubset(keys), \
        str(keys) + ' does not match SMPL keys'
    body_model_mode = 'smpl'
    if {'left_hand_pose', 'right_hand_pose'}.issubset(keys):
        # 'expression', 'jaw_pose', 'leye_pose', 'reye_pose'
        body_model_mode = 'smplx'
    kwargs = {'flat_hand_mean': True, 'use_face_contour': True, 'use_pca': False, 'num_betas': 10,
              'optim_j_regressor': False, 'num_pca_comps': 24}
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = smplx.create('../body_models',
                              # os.getcwd(), # r'\\wsl$\Ubuntu-20.04\home\weichen\zoehuman\mmhuman3d\
                              # data\body_models', #os.getcwd(),
                              model_type=body_model_mode, **kwargs).to(
        torch_device)  # , gender='neutral', num_betas=10)

    # for key in ['global_orient', 'transl', 'body_pose', 'betas']:
    #     print(torch.tensor(param[key][None, frame_idx]).shape)

    if body_model_mode == 'smpl':
        output = body_model(
            global_orient=torch.tensor(param['global_orient'][None, frame_idx], device=torch_device),
            body_pose=torch.tensor(param['body_pose'][None, frame_idx], device=torch_device),
            transl=torch.tensor(param['transl'][None, frame_idx], device=torch_device),
            betas=torch.tensor(param['betas'][None, frame_idx], device=torch_device),
            return_verts=True)
    elif body_model_mode == 'smplx':
        '''for key in ['global_orient', 'body_pose', 'transl', 'betas', 'left_hand_pose']:
            print(param[key][None, frame_idx].shape)'''
        param = revert_smplx_hands_pca(param, num_pca_comps=24)

        output = body_model(
            global_orient=torch.tensor(param['global_orient'][None, frame_idx], device=torch_device),
            body_pose=torch.tensor(param['body_pose'][None, frame_idx], device=torch_device),
            transl=torch.tensor(param['transl'][None, frame_idx], device=torch_device),
            betas=torch.tensor(param['betas'][None, frame_idx], device=torch_device),
            # hand_pose=torch.tensor(param['left_hand_pose'][None, frame_idx], device=torch_device),
            left_hand_pose=torch.tensor(param['left_hand_pose'][None, frame_idx], device=torch_device),
            right_hand_pose=torch.tensor(param['right_hand_pose'][None, frame_idx], device=torch_device),
            # expression=torch.tensor(param['expression'][None, frame_idx], device=torch_device),
            # jaw_pose=torch.tensor(param['jaw_pose'][None, frame_idx], device=torch_device),
            # leye_pose=torch.tensor(param['leye_pose'][None, frame_idx], device=torch_device),
            # reye_pose=torch.tensor(param['reye_pose'][None, frame_idx], device=torch_device),
            return_verts=True)
    else:
        assert False, print('Unsupported body model')

    vertices_vs = output.vertices.detach().cpu().numpy().squeeze()
    trimesh_mesh = trimesh.Trimesh(vertices_vs, body_model.faces, process=False)
    body_model_mesh = trimesh_mesh.as_open3d
    body_model_mesh.compute_vertex_normals()
    # body_model_mesh.paint_uniform_color([1.0, 0.0, 0.0])

    vis_smplx_kp, vs_kps = False, []
    if vis_smplx_kp:
        kps = output.joints.detach().cpu().numpy().squeeze()
        for vs in kps:
            vs_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(vs)
            vs_sphere.paint_uniform_color([1., 0., 0.])
            vs_kps.append(vs_sphere)
        return [body_model_mesh] + vs_kps

    return [body_model_mesh]


def get_ptc_bbox(sample_dir, specific_frame=None, specific_ped_id=None):
    item_list = []
    files = os.listdir(sample_dir)

    for file in files:
        if file.endswith('.pcd'):
            item_list.append(o3d.io.read_point_cloud(os.path.join(sample_dir, file)))
        elif file.endswith('.ply'):
            item_list.append(o3d.io.read_line_set(os.path.join(sample_dir, file)))

    return item_list


if __name__ == '__main__':
    smpl_sample = r'C:\Users\12595\Desktop\GTA-test-data\smpl_2d\smpl_2d_fitting\seq_00007522_730130.npz'
    smplx_sample = r'C:\Users\12595\Desktop\GTA-test-data\smplx\smplx_fitting\seq_00095625_725274.npz'
    smpl_2d_dir = r'C:\Users\12595\Desktop\GTA-test-data\smpl_2d\smpl_2d_fitting'
    smplx_dir = r'C:\Users\12595\Desktop\GTA-test-data\smplx\gta_human2\annotations_multiple_person'
    mta_path = r'E:\gtahuman2_multiple'

    vis_list = []
    seqs_ped = os.listdir(smplx_dir)
    seqs = os.listdir(mta_path)
    seq, frame_idx = 'seq_00087573', 5
    # seq = seqs[0]
    vis_list += process_sequence_ptc(mta_path=mta_path, seq=seq, specific_frame=frame_idx, generate='11fv')
    for file in seqs_ped:
        if file.startswith(seq):
            vis_list += create_smpl_o3d_obj(param_path=os.path.join(smplx_dir, file), frame_idx=frame_idx)
            # vis_list += create_kp_diff_lines(param_path=os.path.join(smplx_dir, file), frame_idx=frame_idx)

    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.base_color = np.array([1, 1, 1, .5])
    o3d.visualization.draw_geometries(vis_list)
    # pass

    # smplx_tianrui = r'\\wsl$\Ubuntu-20.04\home\weichen\zoehuman\tianrui\results'
    # frame = 200
    # obj = []
    # # obj += create_smpl_o3d_obj(param_path=os.path.join(smplx_tianrui, 'exp_debug.npz'), frame_idx=frame)
    # obj += create_kp_diff_lines(param_path=os.path.join(smplx_tianrui, 'exp_debug.npz'), frame_idx=frame)
    # o3d.visualization.draw_geometries(obj)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys

sys.path.insert(1, 'models')
sys.path.insert(1, '')

from os.path import isfile
import torch
from sklearn.manifold import TSNE

from visualizer import draw_smpl
from tqdm import tqdm
import json

import open3d as o3d
import smplx
import trimesh
from collections import Counter
import cv2
from os import listdir, makedirs
from os.path import basename, splitext, join, isdir
import pyrender
from pyrender.viewer import DirectionalLight, Node
import pickle


def rodrigues(rot_vecs, epsilon=1e-8, verpose_return=False):
    """ref: batch_rodrigues() in lbs.py """

    rot_vecs = torch.Tensor(rot_vecs).unsqueeze(0)

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = rot_mat.numpy().reshape(3, 3)

    if verpose_return:
        return rot_mat, rot_dir, angle
    else:
        return rot_mat


def get_cam_coord(global_orient_frame):
    rot = rodrigues(global_orient_frame)

    tf_s2c = np.eye(4)
    tf_s2c[:3, :3] = rot

    # get cam to smpl transformation
    tf_c2s = np.linalg.inv(tf_s2c)

    # assume smpl is at z=1 in cam coord
    p_cam = np.array([0, 0, 1, 1]).reshape((4, 1))
    p_smpl = tf_c2s @ p_cam

    # get cam pos in smpl coord
    offset = p_smpl[:3].squeeze()
    cam_pos = -offset
    return cam_pos


def random_sample(global_orient, sample_num=-1):
    if sample_num == -1:
        return global_orient

    total_num = len(global_orient)
    perm_list = np.random.permutation(total_num)
    sample_idxs = perm_list[:sample_num]
    global_orient = global_orient[sample_idxs]
    return global_orient


def compute_azimuth_elevation(global_orient_frame):
    cam_coord = get_cam_coord(global_orient_frame)
    x, y, z = cam_coord.squeeze().tolist()
    azimuth = np.arctan(x / z)
    elevation = np.arctan(y / (x ** 2 + z ** 2) ** 0.5)
    return azimuth, elevation


def visualize_azimuth_elevation(sample_num=-1):
    content_gta = np.load('data/dataset_extras/gta_smpl_20141.npz')
    content_3dpw = np.load('data/dataset_extras/3dpw_test.npz')
    content_h36m = np.load('data/dataset_extras/h36m_train.npz')
    content_mpi = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')

    global_orient_gta = content_gta['pose'][:, :3]
    global_orient_3dpw = content_3dpw['pose'][:, :3]
    global_orient_h36m = content_h36m['pose'][:, :3]
    global_orient_mpi = content_mpi['pose'][content_mpi['has_smpl'].astype(np.bool), :3]

    global_orient_h36m = adjust_h36m(global_orient_h36m, visualize=False, pathname='global_orient_h36m_adjusted.npy')

    def get_azimuth_elevation(global_orient, sample_num):
        global_orient = random_sample(global_orient, sample_num)

        azimuths, elevations = [], []
        for i in tqdm(range(len(global_orient))):
            global_orient_frame = global_orient[i]
            azimuth, elevation = compute_azimuth_elevation(global_orient_frame)
            azimuths.append(azimuth)
            elevations.append(elevation)

        return azimuths, elevations

    azi_gta, ele_gta = get_azimuth_elevation(global_orient_gta, sample_num=sample_num)
    azi_3dpw, ele_3dpw = get_azimuth_elevation(global_orient_3dpw, sample_num=sample_num)
    azi_h36m, ele_h36m = get_azimuth_elevation(global_orient_h36m, sample_num=sample_num)
    azi_mpi, ele_mpi = get_azimuth_elevation(global_orient_mpi, sample_num=sample_num)

    # Azimuth
    fig, axs = plt.subplots(1, 1)
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs.hist(azi_gta, bins=100, alpha=1, label='GTA-Human', linewidth=3, density=True, histtype='step')
    axs.hist(azi_h36m, bins=100, alpha=0.5, label='Human3.6M', linewidth=3, density=True, histtype='step')
    axs.hist(azi_mpi, bins=100, alpha=0.5, label='MPI-INF-3DHP', linewidth=3, density=True, histtype='step')
    axs.hist(azi_3dpw, bins=100, alpha=0.5, label='3DPW', linewidth=3, density=True, histtype='step')
    # axs.legend(loc='upper right', fontsize=14)
    plt.savefig('/home/caizhongang/Pictures/azimuth.png')
    plt.show()

    # Elevation
    fig, axs = plt.subplots(1, 1)
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs.hist(ele_gta, bins=100, alpha=1, label='GTA-Human', linewidth=3, density=True, histtype='step')
    axs.hist(ele_h36m, bins=100, alpha=0.5, label='Human3.6M', linewidth=3, density=True, histtype='step')
    axs.hist(ele_mpi, bins=100, alpha=0.5, label='MPI-INF-3DHP', linewidth=3, density=True, histtype='step')
    axs.hist(ele_3dpw, bins=100, alpha=0.5, label='3DPW', linewidth=3, density=True, histtype='step')
    axs.legend(loc='upper left', fontsize=14)
    plt.savefig('/home/caizhongang/Pictures/elevation.png')
    plt.show()

    # fig, axs = plt.subplots(2, 2)
    #
    # axs[0, 0].set_title('azimuth')
    # axs[0, 0].set_xlabel('angle (rad)')
    # axs[0, 0].set_ylabel('density')
    #
    # axs[0, 1].set_title('elevation')
    # axs[0, 1].set_xlabel('angle (rad)')
    # axs[0, 1].set_ylabel('density')
    #
    # draw_azimuth_elevation(global_orient_3dpw, axs[0, 0], axs[0, 1], '3DPW', sample_num=sample_num, density=True)
    # draw_azimuth_elevation(global_orient_h36m, axs[0, 0], axs[0, 1], 'Human3.6M', sample_num=sample_num, density=True)
    # draw_azimuth_elevation(global_orient_mpi, axs[0, 0], axs[0, 1], 'MPI-INF-3DHP', sample_num=sample_num, density=True)
    # draw_azimuth_elevation(global_orient_gta, axs[0, 0], axs[0, 1], 'GTA-Human', sample_num=sample_num, density=True)
    #
    # axs[0, 0].legend(loc='upper right')
    # axs[0, 1].legend(loc='upper right')
    #
    # axs[1, 0].set_title('azimuth')
    # axs[1, 0].set_xlabel('angle (rad)')
    # axs[1, 0].set_ylabel('count')
    #
    # axs[1, 1].set_title('elevation')
    # axs[1, 1].set_xlabel('angle (rad)')
    # axs[1, 1].set_ylabel('count')
    #
    # real_azimuth, real_elevation = draw_azimuth_elevation(
    #     np.concatenate([global_orient_3dpw, global_orient_h36m, global_orient_mpi], axis=0),
    #     axs[1, 0], axs[1, 1], 'Real', sample_num=sample_num, density=False)
    # draw_azimuth_elevation(global_orient_gta, axs[1, 0], axs[1, 1], 'GTA-Human', sample_num=sample_num, density=True)
    #
    # axs[1, 0].legend(loc='upper right')
    # axs[1, 1].legend(loc='upper right')
    #
    # plt.show()
    #
    # real_cam_angle_stats = {
    #     'azimuth': {
    #         'n': real_azimuth[0].tolist(),
    #         'bins': real_azimuth[1].tolist()
    #     },
    #     'elevation': {
    #         'n': real_elevation[0].tolist(),
    #         'bins': real_elevation[1].tolist()
    #     }
    # }
    #
    # with open('real_cam_angle_stats.json', 'w') as f:
    #     json.dump(real_cam_angle_stats, f)


def visualize_sphere():
    def draw_cam_coord(global_orient, axs, label):
        global_orient = random_sample(global_orient, sample_num)
        # global_orient = content_gta

        xs, ys, zs = [], [], []
        for i in range(len(global_orient)):
            # for i in range(len(global_orient['pose'])):
            global_orient_frame = global_orient[i]
            # global_orient_frame = global_orient['pose'][i, :3]
            cam_coord = get_cam_coord(global_orient_frame)
            x, y, z = cam_coord.squeeze().tolist()
            xs.append(x)
            ys.append(y)
            zs.append(z)

        axs.scatter(xs, ys, zs, label=label)

    sample_num = 1000

    # draw cam 3D positions (normalized)
    fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))

    axs.set_title('cam angle distribution')
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')

    draw_cam_coord(global_orient_3dpw, axs, label='3dpw')
    draw_cam_coord(global_orient_h36m, axs, label='h36m')
    draw_cam_coord(global_orient_mpi, axs, label='mpi')
    draw_cam_coord(global_orient_gta, axs, label='gta')

    axs.legend(loc='upper right')
    plt.show()


def visualize_axis_angle():
    fig, axs = plt.subplots(2, 3, figsize=(224, 224))
    axs[0, 0].set_title('angle X')
    axs[0, 0].set_xlabel('angle (rad)')
    axs[0, 0].hist(list(global_orient_3dpw[:, 0]), bins=100, alpha=0.5, label='3dpw', density=True)
    axs[0, 0].hist(list(global_orient_h36m[:, 0] - np.pi), bins=100, alpha=0.5, label='h36m (adjusted)', density=True)
    axs[0, 0].hist(list(global_orient_mpi[:, 0]), bins=100, alpha=0.5, label='mpi', density=True)
    axs[0, 0].hist(list(global_orient_gta[:, 0]), bins=100, alpha=0.5, label='gta', density=True)
    axs[0, 0].legend(loc='upper right')

    axs[1, 0].set_title('angle X (unnormalized)')
    axs[1, 0].set_xlabel('angle (rad)')
    axs[1, 0].hist(
        list(global_orient_3dpw[:, 0]) + list(global_orient_h36m[:, 0] - np.pi) + list(global_orient_mpi[:, 0]),
        bins=100, alpha=0.5, label='all others')
    axs[1, 0].hist(list(global_orient_gta[:, 0]), bins=100, alpha=0.5, label='gta')
    axs[1, 0].legend(loc='upper right')

    axs[0, 1].set_title('angle Y')
    axs[0, 1].set_xlabel('angle (rad)')
    axs[0, 1].hist(list(global_orient_3dpw[:, 1]), bins=100, alpha=0.5, label='3dpw', density=True)
    axs[0, 1].hist(list(global_orient_h36m[:, 1]), bins=100, alpha=0.5, label='h36m', density=True)
    axs[0, 1].hist(list(global_orient_mpi[:, 1]), bins=100, alpha=0.5, label='mpi', density=True)
    axs[0, 1].hist(list(global_orient_gta[:, 1]), bins=100, alpha=0.5, label='gta', density=True)
    axs[0, 1].legend(loc='upper right')

    axs[1, 1].set_title('angle Y (unnormalized)')
    axs[1, 1].set_xlabel('angle (rad)')
    axs[1, 1].hist(list(global_orient_3dpw[:, 1]) + list(global_orient_h36m[:, 1]) + list(global_orient_mpi[:, 1]),
                   bins=100, alpha=0.5, label='all others')
    axs[1, 1].hist(list(global_orient_gta[:, 1]), bins=100, alpha=0.5, label='gta')
    axs[1, 1].legend(loc='upper right')

    axs[0, 2].set_title('angle Z')
    axs[0, 2].set_xlabel('angle (rad)')
    axs[0, 2].hist(list(global_orient_3dpw[:, 2]), bins=100, alpha=0.5, label='3dpw', density=True)
    axs[0, 2].hist(list(global_orient_h36m[:, 2]), bins=100, alpha=0.5, label='h36m', density=True)
    axs[0, 2].hist(list(global_orient_mpi[:, 2]), bins=100, alpha=0.5, label='mpi', density=True)
    axs[0, 2].hist(list(global_orient_gta[:, 2]), bins=100, alpha=0.5, label='gta', density=True)
    axs[0, 2].legend(loc='upper right')

    axs[1, 2].set_title('angle Z (unnormalized)')
    axs[1, 2].set_xlabel('angle (rad)')
    axs[1, 2].hist(list(global_orient_3dpw[:, 2]) + list(global_orient_h36m[:, 2]) + list(global_orient_mpi[:, 2]),
                   bins=100, alpha=0.5, label='all others')
    axs[1, 2].hist(list(global_orient_gta[:, 2]), bins=100, alpha=0.5, label='gta')
    axs[1, 2].legend(loc='upper right')

    plt.show()


def adjust_h36m(global_orient_h36m, visualize=False, pathname=None):
    """See visualize_axis_angle for details: h36m has different range: [0, onwards), instead of [-pi, pi]
        if pathname is set, load if exists, and save if doesn't
    """

    def standardize_rotation(angle):
        angle = angle % (2 * np.pi)
        return angle if angle <= np.pi else angle - 2 * np.pi

    def adjust_axis_angle(axis_angle):
        rot, rot_dir, angle = rodrigues(axis_angle, verpose_return=True)
        angle = angle.item()
        angle_adjusted = standardize_rotation(angle)
        axis_angle_adjusted = rot_dir.numpy().squeeze() * angle_adjusted

        rotmat = rodrigues(axis_angle)
        rotmat_adjusted = rodrigues(axis_angle_adjusted)
        assert np.allclose(rotmat, rotmat_adjusted, atol=1e-6)

        return axis_angle_adjusted

    # def adjust_axis_angle(axis_angle):
    #     rotmat = rodrigues(axis_angle)
    #
    #     """ref: SPIN/train/trainer.py. 4x3 rotation matrix is weird!"""
    #     row = np.array([0, 0, 1]).reshape(3, 1)
    #     rotmat_hom = np.concatenate([rotmat, row], axis=-1).astype(np.float32)
    #     axis_angle_adjusted = rotation_matrix_to_angle_axis(torch.tensor(rotmat_hom, dtype=torch.float32).unsqueeze(0))
    #     axis_angle_adjusted = axis_angle_adjusted.numpy().squeeze()
    #
    #     rotmat_adjusted = rodrigues(axis_angle_adjusted)
    #
    #     assert np.allclose(rotmat, rotmat_adjusted, atol=1e-6)
    #     return axis_angle_adjusted

    if pathname is not None and isfile(pathname):
        print('Load processed file:', pathname)
        return np.load(pathname)

    global_orient_h36m_adjusted = np.zeros(global_orient_h36m.shape)
    for i in tqdm(range(len(global_orient_h36m))):
        global_orient_frame = global_orient_h36m[i]
        global_orient_frame_adjusted = adjust_axis_angle(global_orient_frame)
        global_orient_h36m_adjusted[i] = global_orient_frame_adjusted

    if visualize:

        idxs = np.where(global_orient_h36m[:, 0] > 4)[0]

        for i in tqdm(idxs):
            global_orient_frame_extreme = global_orient_h36m[i]
            global_orient_frame_adjusted = global_orient_h36m_adjusted[i]

            pose = torch.zeros((72))
            shape = torch.zeros((10))

            print(global_orient_frame_extreme)
            pose[:3] = torch.Tensor(global_orient_frame_extreme)
            draw_smpl(pose, shape, gender=-1)

            print(global_orient_frame_adjusted)
            pose[:3] = torch.Tensor(global_orient_frame_adjusted)
            draw_smpl(pose, shape, gender=-1)

            # input n to see next
            key = input('input n for next frame: ')
            if key != 'n':
                print(key)
                break

        fig, axs = plt.subplots(1, 3, figsize=(224, 224))
        axs[0].set_title('angle X')
        axs[0].set_xlabel('angle (rad)')
        # axs[0].hist(list(global_orient_3dpw[:, 0]), bins=100, alpha=0.5, label='3dpw', density=True)
        axs[0].hist(list(global_orient_h36m[:, 0]), bins=100, alpha=0.5, label='h36m', density=True)
        axs[0].hist(list(global_orient_h36m_adjusted[:, 0]), bins=100, alpha=0.5, label='h36m (adjusted)', density=True)
        # axs[0].hist(list(global_orient_gta[:, 0]), bins=100, alpha=0.5, label='gta', density=True)
        axs[0].legend(loc='upper right')

        axs[1].set_title('angle Y')
        axs[1].set_xlabel('angle (rad)')
        # axs[1].hist(list(global_orient_3dpw[:, 1]), bins=100, alpha=0.5, label='3dpw', density=True)
        axs[1].hist(list(global_orient_h36m[:, 1]), bins=100, alpha=0.5, label='h36m', density=True)
        axs[1].hist(list(global_orient_h36m_adjusted[:, 1]), bins=100, alpha=0.5, label='h36m (adjusted)', density=True)
        # axs[1].hist(list(global_orient_gta[:, 1]), bins=100, alpha=0.5, label='gta', density=True)
        axs[1].legend(loc='upper right')

        axs[2].set_title('angle Z')
        axs[2].set_xlabel('angle (rad)')
        # axs[2].hist(list(global_orient_3dpw[:, 2]), bins=100, alpha=0.5, label='3dpw', density=True)
        axs[2].hist(list(global_orient_h36m[:, 2]), bins=100, alpha=0.5, label='h36m', density=True)
        axs[2].hist(list(global_orient_h36m_adjusted[:, 2]), bins=100, alpha=0.5, label='h36m (adjusted)', density=True)
        # axs[2].hist(list(global_orient_gta[:, 2]), bins=100, alpha=0.5, label='gta', density=True)
        axs[2].legend(loc='upper right')

        plt.show()

    if pathname is not None and not isfile(pathname):
        print('Save processed h36m adjusted:', pathname)
        np.save(pathname, global_orient_h36m_adjusted)

    return global_orient_h36m_adjusted


def get_joint_array_one(pose_frame, smpl, use_global_orient=False):
    out_batch = smpl(betas=torch.zeros((len(pose_frame), 10)),
                     body_pose=torch.Tensor(pose_frame[:, 3:]),
                     global_orient=
                     torch.Tensor(pose_frame[:, :3])
                     if use_global_orient else
                     torch.zeros((len(pose_frame), 3)))
    joints_batch = out_batch.joints.detach().numpy().reshape((len(pose_frame), -1))
    joints_batch = joints_batch[:, :24 * 3]  # use the original joints. Note: must use the original SMPL
    return joints_batch


def get_joint_array(pose_frame, sample_num, smpl, batch_size, use_global_orient=False):
    pose_frame = random_sample(pose_frame, sample_num)

    # split into batches otherwise OOM
    num_batch = len(pose_frame) // batch_size
    joints = np.zeros((num_batch * batch_size, 24 * 3))
    for i in tqdm(range(num_batch)):
        joints_batch = get_joint_array_one(pose_frame[i * batch_size:(i + 1) * batch_size, :], smpl,
                                           use_global_orient=use_global_orient)
        joints[i * batch_size:(i + 1) * batch_size, :] = joints_batch
    return joints


def draw_embedded(joints_embedded, axs, label, marker='o', alpha=1.0, color=None):
    xs, ys = joints_embedded[:, 0].squeeze().tolist(), joints_embedded[:, 1].squeeze().tolist()
    axs.scatter(xs, ys, label=label, marker=marker, alpha=alpha, color=color)


def visualize_pose_distribution(save_pca=False, save_seq_list=False, draw_seq_list=False, use_global_orient=True,
                                save_gta_embedded=False):
    pca_save_pathname = 'pca.npy'
    seq_list_save_pathname = 'seq_list.json'

    # for sequence visualization
    gta_embedded_save_pathname = 'gta_embedded.npy'
    seqs_embedded_save_pathname = 'seqs_embedded.npz'

    # must include these gta sequences
    selected_seq_idxs = [10411, 29182, 33945, 35679, 44498]
    selected_seq_idxs = set(['seq_{:08d}'.format(i) for i in selected_seq_idxs])

    content_gta = np.load('data/dataset_extras/gta_smpl_20141.npz')
    content_3dpw = np.load('data/dataset_extras/3dpw_test.npz')
    content_h36m = np.load('data/dataset_extras/h36m_train.npz')
    content_mpi = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')

    pose_gta, pose_3dpw, pose_h36m, pose_mpi = \
        content_gta['pose'], content_3dpw['pose'], content_h36m['pose'], \
        content_mpi['pose'][content_mpi['has_smpl'].astype(np.bool), :]

    batch_size = 100
    # smpl = SMPL('data/smpl')
    smpl = smplx.create('/home/caizhongang/github/smplx/models',
                        model_type='smpl',
                        gender='neutral',
                        num_betas=10)

    # sample the joints_gta and corresponding imgnames
    total_num = len(pose_gta)
    perm_list = np.random.permutation(total_num)
    sample_idxs = perm_list[:max([len(pose_gta) // 500, 100])]

    # add additional gta sequences in it
    imgnames = content_gta['imgname']

    selected_seq_idx_example_idxs = {
        k: []
        for k in selected_seq_idxs
    }
    for i, imgname in enumerate(imgnames):
        seq_idx = imgname.split('/')[-2]
        if seq_idx in selected_seq_idxs:
            selected_seq_idx_example_idxs[seq_idx].append(i)

    additional_sample_idxs = []
    for seq_idx, idx_list in selected_seq_idx_example_idxs.items():
        additional_sample_idxs += idx_list

    sample_idxs = np.concatenate([np.array(additional_sample_idxs), sample_idxs])

    pose_gta = pose_gta[sample_idxs]
    imgnames = imgnames[sample_idxs]

    joints_gta = get_joint_array(pose_gta, sample_num=-1, smpl=smpl, batch_size=batch_size,
                                 use_global_orient=use_global_orient)  # -1: already sampled above
    joints_3dpw = get_joint_array(pose_3dpw, sample_num=max([len(pose_3dpw) // 500, 100]), smpl=smpl,
                                  batch_size=batch_size, use_global_orient=use_global_orient)
    joints_h36m = get_joint_array(pose_h36m, sample_num=max([len(pose_h36m) // 500, 100]), smpl=smpl,
                                  batch_size=batch_size, use_global_orient=use_global_orient)
    joints_mpi = get_joint_array(pose_mpi, sample_num=max([len(pose_mpi) // 500, 100]), smpl=smpl,
                                 batch_size=batch_size, use_global_orient=use_global_orient)

    joints = np.vstack([joints_gta, joints_3dpw, joints_h36m, joints_mpi])

    # joints_embedded_tsne = TSNE(n_components=2).fit_transform(joints)

    # pca = PCA(n_components=2).fit(joints)
    pca = PCA(n_components=2).fit(joints[len(additional_sample_idxs):])  # ignore additional idxs
    joints_embedded_pca = pca.transform(joints)

    len_gta, len_3dpw, len_h36m, len_mpi = len(joints_gta), len(joints_3dpw), len(joints_h36m), len(joints_mpi)
    # fig, axs = plt.subplots(1, 2)
    #
    # axs[0].set_title('T-SNE')
    # draw_embedded(joints_embedded_tsne[: len_gta], axs[0], 'gta')
    # draw_embedded(joints_embedded_tsne[len_gta                   : len_gta+len_3dpw], axs[0], '3dpw')
    # draw_embedded(joints_embedded_tsne[len_gta+len_3dpw          : len_gta+len_3dpw+len_h36m], axs[0], 'h36m')
    # draw_embedded(joints_embedded_tsne[len_gta+len_3dpw+len_h36m : len_gta+len_3dpw+len_h36m+len_mpi], axs[0], 'mpi')
    #
    # axs[0].legend(loc='upper right')
    #
    # axs[1].set_title('PCA')
    # draw_embedded(joints_embedded_pca[: len_gta], axs[1], 'gta')
    # draw_embedded(joints_embedded_pca[len_gta                   : len_gta+len_3dpw], axs[1], '3dpw')
    # draw_embedded(joints_embedded_pca[len_gta+len_3dpw          : len_gta+len_3dpw+len_h36m], axs[1], 'h36m')
    # draw_embedded(joints_embedded_pca[len_gta+len_3dpw+len_h36m : len_gta+len_3dpw+len_h36m+len_mpi], axs[1], 'mpi')
    #
    # axs[1].legend(loc='upper right')

    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Pose Distribution (Reduced Dimension)', fontsize=20)
    # axs.set_xlabel('Dimension 1', fontsize=18)
    # axs.set_ylabel('Dimension 2', fontsize=18)
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    draw_embedded(joints_embedded_pca[: len_gta], axs, 'GTA-Human', marker='o')
    draw_embedded(joints_embedded_pca[len_gta + len_3dpw: len_gta + len_3dpw + len_h36m], axs, 'Human3.6M', marker='^')
    draw_embedded(joints_embedded_pca[len_gta + len_3dpw + len_h36m: len_gta + len_3dpw + len_h36m + len_mpi], axs,
                  'MPI-INF-3DHP', marker='s')
    draw_embedded(joints_embedded_pca[len_gta: len_gta + len_3dpw], axs, '3DPW', marker='D')

    # draw_embedded(joints_embedded_pca[len_gta: ], axs, 'Real')
    # draw_embedded(joints_embedded_pca[: len_gta], axs, 'GTA-SMPL')

    axs.legend(loc='upper right', fontsize=10)
    plt.savefig('/home/caizhongang/Pictures/pose_dist.png')
    plt.show()

    components = pca.components_
    if save_pca:
        np.save(pca_save_pathname, components)

    if save_gta_embedded:
        gta_embedded = joints_embedded_pca[: len_gta]
        np.save(gta_embedded_save_pathname, gta_embedded)
        print(gta_embedded_save_pathname, 'saved.')

        # save sequence embedded
        seqs_embedded = {}
        start = 0
        for seq_idx, idx_list in selected_seq_idx_example_idxs.items():
            seqs_embedded[seq_idx] = gta_embedded[start: start + len(idx_list)]
            start += len(idx_list)
        np.savez(seqs_embedded_save_pathname, **seqs_embedded)
        print(seqs_embedded_save_pathname, 'saved.')

    if save_seq_list:
        # sample many curves
        sample_num = 50
        gta_embeddeds = joints_embedded_pca[: len_gta]
        gta_imgnames = imgnames[:len_gta]
        dist_embed_imgnames = []
        for gta_embedded, gta_imgname in zip(gta_embeddeds, gta_imgnames):
            dist = np.linalg.norm(gta_embedded)
            dist_embed_imgnames.append((dist, gta_embedded, gta_imgname))
        dist_embed_imgnames.sort()

        interval = len_gta // sample_num
        selected_dist_embed_imgnames = [dist_embed_imgnames[i * interval] for i in range(sample_num)]

        if draw_seq_list:
            fig, axs = plt.subplots(1, 1)
            axs.set_title('Pose Curves')
            draw_embedded(joints_embedded_pca[: len_gta], axs, 'gta', marker='o')
            for dist, embed, imgname in selected_dist_embed_imgnames:
                seq_idx, _ = splitext(basename(imgname))
                axs.scatter(embed[0], embed[1], label=seq_idx)
            axs.legend(loc='upper right')
            plt.show()

        seq_list = []
        for dist, embed, imgname in selected_dist_embed_imgnames:
            seq_list.append(imgname.split('/')[1])
        with open(seq_list_save_pathname, 'w') as f:
            json.dump(seq_list, f)


def draw_location_map():
    # load stats
    stats = dict(np.load('stats.npz', allow_pickle=True))

    selected_seq_idxs = [22166, 24661, 31981, 33149, 39711, 42020, 47596, 47914]
    selected_seq_idxs = set(['seq_{:08d}'.format(i) for i in selected_seq_idxs])

    color_map = {
        'wild': (102, 204, 0),
        'city_street': (204, 0, 0),
        'roof': (96, 96, 96),
        'special': (102, 0, 204),
        'indoor': (204, 0, 102),
        'city_non_street': (204, 204, 0),
        'coast': (0, 204, 204)
    }

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img = cv2.imread('GTAV-HD-MAP-satellite.jpg')
    location_grids = np.zeros(img.shape[:2], dtype=np.int)  # many samples at the same place

    location_tags = []
    for seq_idx, seq_dict in stats.items():
        seq_dict = seq_dict.item()
        x, y, z = seq_dict['location']
        location_tag = seq_dict['location_tag']
        location_tags.append(location_tag)
        color = color_map[location_tag]
        jpg_x = int(0.6566 * float(x) + 3756)
        jpg_y = int(-0.6585 * float(y) + 5525)
        location_grids[jpg_x, jpg_y] += 1
        radius = max(int(np.log2(location_grids[jpg_x, jpg_y])), 1) * 10
        # img = cv2.circle(img, (jpg_x, jpg_y), radius=radius, color=color, thickness=-1)

        if seq_idx in selected_seq_idxs:
            cv2.putText(img, seq_idx, (jpg_x, jpg_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 255), thickness=1)

    print(Counter(location_tags))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.imwrite('GTAV-location-map-selected.jpg', img)


def visualize_cam_3d():
    global_orient_gta = np.load('data/dataset_extras/gta_smpl_20141.npz')['pose'][:, :3]
    global_orient_h36m = np.load('data/dataset_extras/h36m_train.npz')['pose'][:, :3]
    content_mpi = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')
    global_orient_mpi = content_mpi['pose'][content_mpi['has_smpl'].astype(np.bool), :3]
    global_orient_3dpw = np.load('data/dataset_extras/3dpw_test.npz')['pose'][:, :3]

    def compute_pcd(global_orient, color):
        global_orient = random_sample(global_orient, max(len(global_orient) // 500, 100))

        cam_coords = []
        for global_orient_frame in tqdm(global_orient):
            cam_coord = get_cam_coord(global_orient_frame)
            cam_coords.append(cam_coord)
        cam_coords = np.array(cam_coords)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_coords)
        pcd.colors = o3d.utility.Vector3dVector(color[np.newaxis, :].repeat(repeats=len(cam_coords), axis=0))

        return pcd

    cams_3d_gta = compute_pcd(global_orient_gta, np.array([22 / 255, 114 / 255, 177 / 255]))
    cams_3d_h36m = compute_pcd(global_orient_h36m, np.array([255 / 255, 127 / 255, 14 / 255]))
    cams_3d_mpi = compute_pcd(global_orient_mpi, np.array([44 / 255, 160 / 255, 44 / 255]))
    cams_3d_3dpw = compute_pcd(global_orient_3dpw, np.array([214 / 255, 39 / 255, 40 / 255]))

    # Obtain mesh
    smpl = smplx.create('/home/caizhongang/github/smplx/models',
                        model_type='smpl',
                        gender='neutral',
                        num_betas=10)
    model_output = smpl(return_verts=True)  # T-pose
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    faces = smpl.faces
    trimesh_mesh = trimesh.Trimesh(vertices, faces, process=False)
    open3d_mesh = trimesh_mesh.as_open3d
    open3d_mesh.compute_vertex_normals()

    # Open3D scene
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    visual = [axis, open3d_mesh, cams_3d_gta, cams_3d_h36m, cams_3d_mpi, cams_3d_3dpw]
    o3d.visualization.draw_geometries(visual)


def evaluate_cam_angles():
    with open('results.json', 'r') as f:
        results = json.load(f)

    content_gta = np.load('data/dataset_extras/gta_smpl_20141.npz')
    global_orient_gta = content_gta['pose'][:, :3]
    imgname = content_gta['imgname']
    sample_num = len(global_orient_gta)

    total_num = len(global_orient_gta)
    perm_list = np.random.permutation(total_num)
    sample_idxs = perm_list[:sample_num]

    global_orient_gta = global_orient_gta[sample_idxs]
    imgnames = imgname[sample_idxs]

    prefix = '/mnt/lustre/share/DSK/datasets/GTA_Auto_Gen/dataset'

    if isfile('cam_angles.json'):
        print('Found cam_angles.json. Loading...')
        with open('cam_angles.json', 'r') as f:
            content = json.load(f)
        azimuth_results_loaded = content['azimuth_results']
        azimuth_list = content['azimuth_list']
        elevation_results_loaded = content['elevation_results']
        elevation_list = content['elevation_list']

        # need to convert all keys from str to float
        azimuth_results = {
            float(k): v
            for k, v in azimuth_results_loaded.items()
        }

        elevation_results = {
            float(k): v
            for k, v in elevation_results_loaded.items()
        }

    else:
        azimuth_results = {}
        azimuth_list = []
        elevation_results = {}
        elevation_list = []
        for example_idx, global_orient_frame in enumerate(tqdm(global_orient_gta)):
            imgname = imgnames[example_idx]
            key = join(prefix, imgname)
            mpjpe, pa_mpjpe = results[key]

            azimuth, elevation = compute_azimuth_elevation(global_orient_frame)

            if azimuth not in azimuth_results:
                azimuth_results[azimuth] = {
                    'mpjpes': [],
                    'pa_mpjpes': []
                }

            if elevation not in elevation_results:
                elevation_results[elevation] = {
                    'mpjpes': [],
                    'pa_mpjpes': []
                }

            azimuth_results[azimuth]['mpjpes'].append(mpjpe)
            azimuth_results[azimuth]['pa_mpjpes'].append(pa_mpjpe)

            elevation_results[elevation]['mpjpes'].append(mpjpe)
            elevation_results[elevation]['pa_mpjpes'].append(pa_mpjpe)

            azimuth_list.append(azimuth)
            elevation_list.append(elevation)

        content = {
            'azimuth_results': azimuth_results,
            'azimuth_list': azimuth_list,
            'elevation_results': elevation_results,
            'elevation_list': elevation_list
        }
        with open('cam_angles.json', 'w') as f:
            json.dump(content, f)
        print('cam_angles.json saved.')

    azi_x, azi_y_mpjpe, azi_y_pa_mpjpe = process_by_example(azimuth_results, min_num=0)
    ele_x, ele_y_mpjpe, ele_y_pa_mpjpe = process_by_example(elevation_results, min_num=0)

    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Azimuth')
    # axs.set_ylabel('Error')
    # axs.set_xlabel('Angle (Radian)')
    axs2 = axs.twinx()
    # axs2.set_ylabel('Density')

    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs2.tick_params(axis='y', labelsize=14)

    hist_ret = axs2.hist(azimuth_list, bins=100, label='Data', alpha=0.5, density=True)

    azi_x_mpjpe, azi_y_mpjpe = smooth_by_bins(azi_x, azi_y_mpjpe, hist_ret)
    azi_x_pa_mpjpe, azi_y_pa_mpjpe = smooth_by_bins(azi_x, azi_y_pa_mpjpe, hist_ret)

    axs.plot(azi_x_mpjpe, azi_y_mpjpe, label='MPJPE', linewidth=3, color='g')
    axs.plot(azi_x_pa_mpjpe, azi_y_pa_mpjpe, label='PA-MPJPE', linewidth=3, color='orange')
    # axs.legend(loc='upper right')
    plt.savefig('/home/caizhongang/Pictures/azimuth.png')
    fig.show()

    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Elevation')
    # axs.set_ylabel('Error')
    # axs.set_xlabel('Angle (Radian)')
    axs2 = axs.twinx()
    # axs2.set_ylabel('Density')

    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs2.tick_params(axis='y', labelsize=14)

    hist_ret = axs2.hist(elevation_list, bins=100, label='Data', alpha=0.5, density=True)

    ele_x_mpjpe, ele_y_mpjpe = smooth_by_bins(ele_x, ele_y_mpjpe, hist_ret)
    ele_x_pa_mpjpe, ele_y_pa_mpjpe = smooth_by_bins(ele_x, ele_y_pa_mpjpe, hist_ret)

    axs.plot(ele_x_mpjpe, ele_y_mpjpe, label='MPJPE', linewidth=3, color='g')
    axs.plot(ele_x_pa_mpjpe, ele_y_pa_mpjpe, label='PA-MPJPE', linewidth=3, color='orange')
    # axs.legend(loc='upper right')
    plt.savefig('/home/caizhongang/Pictures/elevation.png')
    fig.show()


def process_by_example(results, min_num=1000):
    data = []
    for occ_percentage, metrics in results.items():
        if len(metrics['mpjpes']) < min_num:  # skip if too few sample points
            continue
        mpjpe = sum(metrics['mpjpes']) / len(metrics['mpjpes'])
        pa_mpjpe = sum(metrics['pa_mpjpes']) / len(metrics['pa_mpjpes'])
        data.append((occ_percentage, mpjpe, pa_mpjpe))

    data.sort()  # sort so x is in ascending order
    x, y_mpjpe, y_pa_mpjpe = [], [], []
    for occ_percentage, mpjpe, pa_mpjpe in data:
        x.append(occ_percentage)
        y_mpjpe.append(mpjpe)
        y_pa_mpjpe.append(pa_mpjpe)

    return x, y_mpjpe, y_pa_mpjpe


def evaluate_occlusion():
    occ_min = self_occ_min = 0
    occ_max = self_occ_max = 0.4

    # load stats
    stats = dict(np.load('stats.npz', allow_pickle=True))

    # load results
    with open('new_results.json', 'r') as f:
        results = json.load(f)

    # ref: kp_utils.py from temporal-smplify
    # note: spin3, left/right clavicles have no corresponding GTA joints
    gta_to_smpl = [
        14,  # 00 pelvis (SMPL)     - 14 spine3 (GTA)
        19,  # 01 left hip          - 19 left hip
        16,  # 02 right hip         - 16 right hip
        13,  # 03 spine1            - 13 spine2
        20,  # 04 left knee         - 20 left knee
        17,  # 05 right knee        - 17 right knee
        11,  # 06 spine2            - 11 spine0
        21,  # 07 left ankle        - 21 left ankle
        18,  # 08 right ankle       - 18 right ankle
        # 09 spine3            - no match
        24,  # 10 left foot         - 24 SKEL_L_Toe0
        49,  # 11 right foot        - 49 SKEL_R_Toe0
        2,  # 12 neck              - 02 neck
        # 13 left clavicle     - no match, 07 left clavicle different convention
        # 14 right clavicle    - no match, 03 right clavicle different convention
        1,  # 15 head              - 01 head center
        8,  # 16 left shoulder     - 08 left shoulder
        4,  # 17 right shoulder    - 04 right shoulder
        9,  # 18 left elbow        - 09 left elbow
        5,  # 19 right elbow       - 05 right elbow
        55,  # 20 left wrist        - 55 left wrist
        6,  # 21 right wrist       - 06 right wrist
        57,  # 22 left_hand         - 57 SKEL_L_Finger20 (left middle finger root)
        81,  # 23 right_hand        - 81 SKEL_R_Finger20 (right middle finger root)
    ]

    # Task 1: evaluate occ/self-occ by examples
    occ_results = {}  # key: occ_percentage, val: {list of mpjpes, list of pa_mpjpe}
    self_occ_results = {}  # key: occ_percentage, val: {list of mpjpes, list of pa_mpjpe}
    occ_percentage_list = []
    self_occ_percentage_list = []
    # Task 2: evaluate occ/self-occ by joints
    joint_occ = np.zeros((len(gta_to_smpl)))
    joint_self_occ = np.zeros((len(gta_to_smpl)))
    joint_occ_pa_mpjpe = np.zeros((len(gta_to_smpl)))
    joint_self_occ_pa_mpjpe = np.zeros((len(gta_to_smpl)))
    for imgname, (mpjpe, pa_mpjpe) in results.items():
        imgname_parts = imgname.split('/')
        seq_idx = imgname_parts[-2]  # e.g. 'seq_00000004'
        basename = imgname_parts[-1]  # e.g. '00000000.jpeg'
        stem, _ = splitext(basename)  # e.g. '00000000'
        frame_idx = int(stem)

        seq_stats = stats[seq_idx].item()  # recover dict from numpy object

        # convert from gta joints to smpl joints
        occ = seq_stats['occ'][frame_idx, gta_to_smpl]
        self_occ = seq_stats['self_occ'][frame_idx, gta_to_smpl]

        # Task 1
        occ_percentage = occ.sum() / len(occ)
        self_occ_percentage = self_occ.sum() / len(self_occ)

        if occ_min <= occ_percentage <= occ_max:
            if occ_percentage not in occ_results:
                occ_results[occ_percentage] = {
                    'mpjpes': [],
                    'pa_mpjpes': []
                }

            occ_results[occ_percentage]['mpjpes'].append(mpjpe)
            occ_results[occ_percentage]['pa_mpjpes'].append(pa_mpjpe)
            occ_percentage_list.append(occ_percentage)

        if self_occ_min <= self_occ_percentage <= self_occ_max:
            if self_occ_percentage not in self_occ_results:
                self_occ_results[self_occ_percentage] = {
                    'mpjpes': [],
                    'pa_mpjpes': []
                }

            self_occ_results[self_occ_percentage]['mpjpes'].append(mpjpe)
            self_occ_results[self_occ_percentage]['pa_mpjpes'].append(pa_mpjpe)
            self_occ_percentage_list.append(self_occ_percentage)

        # Task 2
        joint_occ += occ
        joint_self_occ += self_occ
        joint_occ_pa_mpjpe += pa_mpjpe * occ
        joint_self_occ_pa_mpjpe += pa_mpjpe * self_occ

    # Task 1
    occ_x, occ_y_mpjpe, occ_y_pa_mpjpe = process_by_example(occ_results)
    self_occ_x, self_occ_y_mpjpe, self_occ_y_pa_mpjpe = process_by_example(self_occ_results)

    # Draw Task 1
    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Occlusion')
    # axs.set_ylabel('Error')
    # axs.set_xlabel('%')
    axs2 = axs.twinx()
    # axs2.set_ylabel('Density')

    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs2.tick_params(axis='y', labelsize=14)

    axs2.hist(occ_percentage_list, bins=9, label='Data', alpha=0.5, density=True, log=True)
    axs.plot(occ_x, occ_y_mpjpe, label='MPJPE', linewidth=3, color='g')
    axs.plot(occ_x, occ_y_pa_mpjpe, label='PA-MPJPE', linewidth=3, color='orange')
    # axs.legend(loc='top', fontsize=14)

    axs.set_ylim([60, 160])
    plt.savefig('/home/caizhongang/Pictures/occlusion.png')
    fig.show()

    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Self-Occlusion')
    # axs.set_ylabel('Error')
    # axs.set_xlabel('%')
    axs2 = axs.twinx()
    # axs2.set_ylabel('Density')

    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs2.tick_params(axis='y', labelsize=14)

    axs2.hist(self_occ_percentage_list, bins=9, label='Data', alpha=0.5, density=True, log=True)
    axs.plot(self_occ_x, self_occ_y_mpjpe, label='MPJPE', linewidth=3, color='g')
    axs.plot(self_occ_x, self_occ_y_pa_mpjpe, label='PA-MPJPE', linewidth=3, color='orange')
    # axs.legend(loc='top', fontsize=14)

    # axs.set_ylim([40, 180])
    plt.savefig('/home/caizhongang/Pictures/self_occlusion.png')
    fig.show()

    # Task 2
    # def process_by_joint(attribute):
    #     ma = attribute.max()
    #     mi = attribute.min()
    #     normalized = (attribute - mi) / (ma - mi)
    #     return attribute, normalized, ma, mi
    #
    # joint_occ_normalized = process_by_joint(joint_occ)
    # joint_self_occ += self_occ
    # joint_occ_pa_mpjpe += pa_mpjpe * occ
    # joint_self_occ_pa_mpjpe += pa_mpjpe * self_occ
    #
    # fig, axs = plt.subplots(1, 1)


def smooth_by_bins(xs, ys, hist_ret):
    n, bins = hist_ret[0], hist_ret[1]
    mi = bins[0]
    width = bins[1] - bins[0]

    # create xs from mid of bins
    new_xs = []
    new_xy_dict = {}  # key: new_x, val: list of ys
    for bin in bins[:-1]:
        new_x = bin + width / 2
        new_xs.append(new_x)
        new_xy_dict[new_x] = []

    for x, y in zip(xs, ys):
        new_x = new_xs[min(int((x - mi) / width), len(bins) - 2)]
        # -2: highest edge; start at 0
        new_xy_dict[new_x].append(y)

    new_xy = []
    for new_x, ys in new_xy_dict.items():
        if sum(ys) == 0:
            continue
        y_mean = sum(ys) / len(ys)
        new_xy.append((new_x, y_mean))

    new_xy.sort()

    new_xs, new_ys = [], []
    for new_x, new_y in new_xy:
        new_xs.append(new_x)
        new_ys.append(new_y)

    return new_xs, new_ys


def evaluate_pose_dist(t_pose_center=False):
    # load results
    with open('results.json', 'r') as f:
        results = json.load(f)

    content_gta = np.load('data/dataset_extras/gta_smpl_20141.npz')
    pose_gta = content_gta['pose']
    imgname = content_gta['imgname']
    sample_num = len(pose_gta)
    batch_size = 100

    smpl = smplx.create('/home/caizhongang/github/smplx/models',
                        model_type='smpl',
                        gender='neutral',
                        num_betas=10)

    total_num = len(pose_gta)
    perm_list = np.random.permutation(total_num)
    sample_idxs = perm_list[:sample_num]

    pose_gta = pose_gta[sample_idxs]
    imgnames = imgname[sample_idxs]

    if isfile('joints_gta.npy'):
        print('Found processed joints_gta.npy. Loading...')
        joints_gta = np.load('joints_gta.npy')
    else:
        joints_gta = get_joint_array(pose_gta, sample_num=-1,  # already sampled, now use all
                                     smpl=smpl, batch_size=batch_size,
                                     use_global_orient=False)
        np.save('joints_gta.npy', joints_gta)
        print('joints_gta.npy saved.')

    # Consider T-pose as center
    if t_pose_center:
        t_pose_out = smpl()
        joints_gta_mean = t_pose_out.joints.detach().numpy().reshape((45 * 3))
        joints_gta_mean = joints_gta_mean[:24 * 3]

    # Consider distribution center, or (see below)
    else:
        joints_gta_mean = joints_gta.mean(axis=0)

    prefix = '/mnt/lustre/share/DSK/datasets/GTA_Auto_Gen/dataset'

    pose_results = {}
    dist_list = []
    for example_idx, joints_gta_frame in enumerate(joints_gta):
        imgname = imgnames[example_idx]
        key = join(prefix, imgname)
        mpjpe, pa_mpjpe = results[key]

        # compute distance
        dist = np.linalg.norm(joints_gta_frame - joints_gta_mean)

        if dist not in pose_results:
            pose_results[dist] = {
                'mpjpes': [],
                'pa_mpjpes': []
            }

        pose_results[dist]['mpjpes'].append(mpjpe)
        pose_results[dist]['pa_mpjpes'].append(pa_mpjpe)
        dist_list.append(dist)

    dist_x, y_mpjpe, y_pa_mpjpe = process_by_example(pose_results, min_num=0)

    fig, axs = plt.subplots(1, 1)

    if t_pose_center:
        axs.set_xlim([0, 3.0])
        axs.set_ylim([40, 150])
    else:
        axs.set_xlim([0, 2.5])
        axs.set_ylim([40, 150])

    # axs.set_title('Pose Distribution')
    # axs.set_ylabel('Error')
    # axs.set_xlabel('Distance to Center')
    axs2 = axs.twinx()
    # axs2.set_ylabel('Density')

    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs2.tick_params(axis='y', labelsize=14)

    hist_ret = axs2.hist(dist_list, bins=100, label='Data', alpha=0.5, density=True)

    dist_x_mpjpe, y_mpjpe = smooth_by_bins(dist_x, y_mpjpe, hist_ret)
    dist_x_pa_mpjpe, y_pa_mpjpe = smooth_by_bins(dist_x, y_pa_mpjpe, hist_ret)

    axs.plot(dist_x_mpjpe, y_mpjpe, label='MPJPE', linewidth=3, color='g')
    axs.plot(dist_x_pa_mpjpe, y_pa_mpjpe, label='PA-MPJPE', linewidth=3, color='orange')
    # axs.legend(loc='upper right')
    plt.savefig('/home/caizhongang/Pictures/pose_dist_from_' +
                ('t_pose' if t_pose_center else 'mean') +
                '.png')
    fig.show()


# def create_raymond_lights():
#     thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
#     phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
#
#     nodes = []
#
#     for phi, theta in zip(phis, thetas):
#         xp = np.sin(theta) * np.cos(phi)
#         yp = np.sin(theta) * np.sin(phi)
#         zp = np.cos(theta)
#
#         z = np.array([xp, yp, zp])
#         z = z / np.linalg.norm(z)
#         x = np.array([-z[1], z[0], 0.0])
#         if np.linalg.norm(x) == 0:
#             x = np.array([1.0, 0.0, 0.0])
#         x = x / np.linalg.norm(x)
#         y = np.cross(z, x)
#
#         matrix = np.eye(4)
#         matrix[:3, :3] = np.c_[x, y, z]
#         nodes.append(Node(
#             light=DirectionalLight(color=np.ones(3), intensity=1.0),
#             matrix=matrix
#         ))
#
#     return nodes

def create_raymond_lights():
    # set directional light at axis origin, with -z direction align with +z direction of camera/world frame
    matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return [Node(light=DirectionalLight(color=np.ones(3), intensity=2.0),
                 matrix=matrix)]


def draw_overlay(img,
                 body_model,
                 camera, cam_poses, H, W,
                 visualize=False,
                 **kwargs):
    for k, v in kwargs.items():
        if len(v.shape) < 2:
            v = np.expand_dims(v, axis=0)
        try:
            kwargs[k] = torch.Tensor(v)
        except:
            print(k, v)
            exit()

    model_output = body_model(return_verts=True, **kwargs)

    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    faces = body_model.faces

    out_mesh = trimesh.Trimesh(vertices, faces, process=False)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(
        out_mesh,
        material=material)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # convert opencv cam pose to opengl cam pose
    # x,-y,-z -> x,y,z
    R_convention = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    cam_poses = R_convention @ cam_poses

    # cam_pose: transformation from cam coord to world coord
    scene.add(camera, pose=cam_poses)

    if visualize:
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                 viewport_size=(W, H),
                                 cull_faces=False,
                                 run_in_thread=False,
                                 registered_keys=dict())

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    output_img = (color[:, :, :-1] * valid_mask +
                  (1 - valid_mask) * img)

    img = (output_img * 255).astype(np.uint8)

    return img


def visualize_pose(skip_exist=True, draw_gta_embedded=False, img_load_dir='seqs/images', pkl_load_dir='seqs/pkls',
                   img_save_dir='seqs/visual'):
    """Overlays smpl onto images; draw the sequence as curves on the pose distribution"""

    camera = pyrender.camera.IntrinsicsCamera(
        fx=1158.0337, fy=1158.0337,
        cx=960, cy=540)

    smpl = smplx.create('D:\GTAV\mta-sample\overlay',
                        model_type='smpl',
                        gender='neutral',
                        num_betas=10)

    seqs = sorted(listdir(img_load_dir))
    for seq in tqdm(seqs):

        # skip processed seqs
        if skip_exist and isdir(join(img_save_dir, seq)):
            continue
        if seq.startswith("seq_00000261"):
            pass
        else: continue

        # load pkl
        pkl_load_pathname = join(pkl_load_dir, seq, seq + '.pkl')
        print("pkl load dir is " + pkl_load_dir)
        print("pkl dir is " + pkl_load_pathname)
        print("seq is " + seq)
        with open(pkl_load_pathname, 'rb') as f:
            content = pickle.load(f, encoding='latin1')

        num_frames = len(content['betas'])
        for frame_idx in tqdm(range(num_frames)):
            img_load_pathname = join(img_load_dir, seq, '{:08d}.jpeg'.format(frame_idx))
            img = cv2.imread(img_load_pathname)

            smpl_img = draw_overlay(img, smpl, camera, cam_poses=np.eye(4), H=1080, W=1920,
                                    visualize=False,
                                    betas=content['betas'][frame_idx],
                                    global_orient=content['global_orient'][frame_idx],
                                    body_pose=content['body_pose'][frame_idx],
                                    transl=content['transl'][frame_idx]
                                    )

            img_save_dir_seq = join(img_save_dir, seq)
            if not isdir(img_save_dir_seq):
                makedirs(img_save_dir_seq)
            img_save_pathname = join(img_save_dir_seq, '{:08d}.jpeg'.format(frame_idx))
            cv2.imwrite(img_save_pathname, smpl_img)

    # draw pose curves on the distribution
    fig, axs = plt.subplots(1, 1)
    # axs.set_title('Pose Sequence')
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)

    def draw_embedded_video(joints_embedded, axs, label, color, keyframes=None):
        xs, ys = joints_embedded[:, 0].squeeze().tolist(), joints_embedded[:, 1].squeeze().tolist()
        axs.plot(xs, ys, label=label, marker='.', color=color, ms=5)
        if keyframes is not None:
            axs.scatter([xs[i] for i in keyframes], [ys[i] for i in keyframes], color=color, marker='D', s=50)

    # load pca
    components_load_pathname = 'pca.npy'
    components = np.load(components_load_pathname)

    selected_seqs = {
        'seq_00010411': ('r', [0, 25, 38]),
        'seq_00029182': ('g', [0, 23, 46]),
        'seq_00033945': ('b', [11, 20, 49]),
        'seq_00035679': ('c', [16, 30, 69]),
        'seq_00044498': ('m', [24, 42, 56]),
    }

    # draw gta poses (reduced dimension)
    if draw_gta_embedded:
        gta_embedded = np.load('gta_embedded.npy')
        draw_embedded(gta_embedded, axs, label='GTA-Human', marker='o', alpha=1, color='gray')

    # draw pose sequence

    if isfile('seqs_embedded.npz'):
        seqs_embedded = np.load('seqs_embedded.npz')
        cnt = 1
        for seq_idx, seq_embedded in seqs_embedded.items():
            color, keyframes = selected_seqs[seq_idx]
            draw_embedded_video(seq_embedded, axs, 'Sequence {}'.format(cnt), color=color, keyframes=keyframes)
            cnt += 1
    else:
        cnt = 1
        for seq in tqdm(seqs):

            if seq not in selected_seqs:
                continue
            color, keyframes = selected_seqs[seq]

            # load pkl
            pkl_load_pathname = join(pkl_load_dir, seq + '.pkl')
            with open(pkl_load_pathname, 'rb') as f:
                content = pickle.load(f, encoding='latin1')

            body_pose = content['body_pose']
            global_orient = content['global_orient']
            pose_gta = np.concatenate([global_orient, body_pose], axis=1)

            joints_gta = get_joint_array_one(pose_gta, smpl=smpl, use_global_orient=True)

            joints_embedded_pca = np.einsum('ij,kj->ki', components, joints_gta)

            draw_embedded_video(joints_embedded_pca, axs, 'Sequence {}'.format(cnt), color=color, keyframes=keyframes)
            cnt += 1

    axs.legend(loc='upper right', fontsize=10)
    plt.savefig('/home/caizhongang/Pictures/pose_sequence.png')
    plt.show()


def draw_data_amount_curve():
    xs = [0, 0.5, 1, 2, 3, 4]
    baseline = [61.7, 61.7, 61.7, 61.7, 61.7, 61.7]
    mpjpes = [98.5, 91.36, 90.19, 91.33, 89.8, 88.76]
    pa_mpjpes = [61.7, 59.59, 58.66, 57.5, 57.23, 56.05]

    fig, axs = plt.subplots(1, 1)
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)

    axs.plot(xs, baseline, label='Baseline', linewidth=4, linestyle='--', color='gray')
    axs.plot(xs, pa_mpjpes, label='GTA-Human', linewidth=4, color='orange')

    # axs.set_xlim([-100, 1500])
    axs.set_ylim([55.5, 62.5])
    axs.legend(loc='lower left', fontsize=16)

    plt.savefig('/home/caizhongang/Pictures/data_amount.png')
    plt.show()


def neurips_2021():
    # content_gta = dict(np.load('data/dataset_extras/gta_smpl_20141.npz'))
    # global_orient_gta = content_gta['pose'][:, :3]

    # draw_location_map()

    # evaluate_occlusion()

    # visualize_pose_distribution(save_pca=True, save_seq_list=False, draw_seq_list=False, save_gta_embedded=True, use_global_orient=True)
    # evaluate_pose_dist(t_pose_center=False)
    # evaluate_pose_dist(t_pose_center=True) # must run separately from the one above
    # visualize_pose(skip_exist=True, draw_gta_embedded=True)

    visualize_azimuth_elevation(sample_num=-1)
    # evaluate_cam_angles()

    # visualize_cam_3d()

    # draw_data_amount_curve()

    pass


def main():
    # content_gta = dict(np.load('data/dataset_extras/gta_smplx.npz'))
    content_gta = dict(np.load('data/dataset_extras/gta_smpl_20141.npz'))

    content_3dpw = dict(np.load('data/dataset_extras/3dpw_test.npz'))
    content_h36m = dict(np.load('data/dataset_extras/h36m_train.npz'))
    content_mpi = dict(np.load('data/dataset_extras/mpi_inf_3dhp_train.npz'))

    global_orient_gta = content_gta['pose'][:, :3]
    global_orient_3dpw = content_3dpw['pose'][:, :3]
    global_orient_h36m = content_h36m['pose'][:, :3]
    global_orient_mpi = content_mpi['pose'][content_mpi['has_smpl'].astype(np.bool), :3]

    global_orient_h36m = adjust_h36m(visualize=False, pathname='global_orient_h36m_adjusted.npy')

    # visualize_axis_angle()
    # visualize_sphere()
    # visualize_azimuth_elevation(sample_num=10000)

    # visualize_pose_distribution(content_gta['pose'],
    #                             content_3dpw['pose'],
    #                             content_h36m['pose'],
    #                             content_mpi['pose'])


if __name__ == '__main__':
    np.random.seed(0)

    # main()
    neurips_2021()

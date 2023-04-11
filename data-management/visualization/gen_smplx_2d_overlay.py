import math
import os
import random

import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import pyrender
import trimesh

import smplx

from pyrender.viewer import DirectionalLight, Node

RECORD_FPS = 15


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
            exit()
        # print(v.shape)
    '''for k, v in kwargs.items():
        print(k, v.shape)'''
    # kwargs['body_pose'] = torch.cat((kwargs['body_pose'][:, :], kwargs['global_orient'][:, :], ), 1)
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

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=1)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    output_img = (color[:, :, :-1] * valid_mask +
                  (1 - valid_mask) * img)

    img = (output_img * 255).astype(np.uint8)

    return img


def parse_summary(pathname):
    with open(pathname, 'r') as f:
        content = f.read().splitlines()
    ped_idx_list = content[0][1:-1].split(',')
    ped_idx_list = [int(x) for x in ped_idx_list]
    return ped_idx_list


def parse_all_ped_ids(csv_file):
    content_csv = pd.read_csv(csv_file)
    content = content_csv.to_dict()
    ped_id_list = list(set(list(content['ped_id'].values())))
    return ped_id_list


def parse_ped_ids_with_seqs(csv_file):
    content_csv = pd.read_csv(csv_file)
    content = content_csv.to_dict()
    ped_id_list = list(set(list(content['ped_id'].values())))
    frame_count = len(list(set(list(content['frame'].values()))))
    [[cam_3dx, cam_3dy, cam_3dz]] = content_csv[['cam_3D_x', 'cam_3D_y', 'cam_3D_z']][:1].values.tolist()
    # print(cam_3dx, cam_3dy, cam_3dz)
    ped_id_list_per_frame = []
    for frame_idx in range(frame_count):
        ped_distance_dict = {}
        for ped_id in ped_id_list:
            distance = 0
            ped_id_coord = content_csv[(content_csv['frame'] == frame_idx) & (content_csv['ped_id'] == ped_id)][
                ['3D_x', '3D_y', '3D_z']].values.tolist()
            for pt in ped_id_coord:
                distance += math.sqrt((cam_3dx - pt[0]) ** 2 + (cam_3dy - pt[1]) ** 2 + (cam_3dz - pt[2]) ** 2)
                pass
            distance = distance / 99
            ped_distance_dict[ped_id] = distance
            # print(ped_id_csv)
        sorted_ped_id = sorted(ped_distance_dict, key=ped_distance_dict.get, reverse=True)
        ped_id_list_per_frame.append(sorted_ped_id)

    return ped_id_list_per_frame


def visualize_pose(skip_exist=False, img_load_dir='seqs/images', npz_load_dir='seqs/pkls',
                   img_save_dir='seqs/visual', torch_device='cuda', body_model_type='smpl', specific_seq=None):
    """Overlays smpl onto images; draw the sequence as curves on the pose distribution"""

    camera = pyrender.camera.IntrinsicsCamera(fx=1158.0337, fy=1158.0337, cx=960, cy=540)
    kwargs = {'flat_hand_mean': True, 'use_face_contour': True, 'use_pca': True, 'num_betas': 10,
              'num_pca_comps': 24, }
    # 'optim_j_regressor': True}
    body_model = smplx.create(r'C:\Users\12595\Desktop\MTA\python\body_models',
                              model_type=body_model_type, **kwargs).to(torch_device)

    seqs = sorted(os.listdir(img_load_dir))
    random.shuffle(seqs)
    if specific_seq:
        seqs = [specific_seq]

    for seq in tqdm.tqdm(seqs):
        print(seq)

        # skip processed seqs
        if skip_exist and os.path.isdir(os.path.join(img_save_dir, seq)):
            continue
        if seq.startswith("seq_") and seq[-3:].isdigit():
            pass
        else:
            continue

        # ped_id_list = parse_all_ped_ids(join(img_load_dir, seq, 'peds.csv'))
        # ped_id_list = parse_all_ped_ids(os.path.join(img_load_dir, seq, 'peds.csv'))
        ped_id_list_per_frame = parse_ped_ids_with_seqs(os.path.join(img_load_dir, seq, 'peds.csv'))
        num_frames = len(ped_id_list_per_frame)
        try:
            for frame_idx in tqdm.tqdm(range(num_frames)):
                img_load_pathname = os.path.join(img_load_dir, seq, '{:08d}.jpeg'.format(frame_idx))
                ped_id_list = ped_id_list_per_frame[frame_idx]
                img = cv2.imread(img_load_pathname)

                for ped_idx in ped_id_list:
                    npz_load_pathname = os.path.join(npz_load_dir, seq + '_' + '{:06d}'.format(ped_idx))
                    try:
                        param = dict(np.load(npz_load_pathname + '.npz', allow_pickle=True))
                    except FileNotFoundError:
                        try:
                            param = dict(np.load(npz_load_pathname + '.pkl', allow_pickle=True))
                        except FileNotFoundError:
                            # print(npz_load_pathname, 'not found')
                            continue
                    # img = np.zeros((8192, 8192, 3), dtype='uint8')
                    # img[:] = 255
                    if body_model_type == 'smpl':
                        img = draw_overlay(img, body_model, camera, cam_poses=np.eye(4), H=1080, W=1920,
                                           visualize=False,
                                           # betas=content['betas'][frame_idx],
                                           # betas=content['betas'][0],
                                           global_orient=torch.tensor(param['global_orient'][None, frame_idx],
                                                                      device=torch_device),
                                           body_pose=torch.tensor(param['body_pose'][None, frame_idx], device=torch_device),
                                           transl=torch.tensor(param['transl'][None, frame_idx], device=torch_device),
                                           betas=torch.tensor(param['betas'][None, frame_idx], device=torch_device), )
                    elif body_model_type == 'smplx':
                        # print(param['transl'][None, frame_idx])
                        # print(param['gt_kp'][frame_idx][0])

                        # for key in ['global_orient', 'transl', 'body_pose', 'betas']:
                        #     print(torch.tensor(param[key][None, frame_idx]).shape)
                        img = draw_overlay(img, body_model, camera, cam_poses=np.eye(4), H=1080, W=1920,
                                           visualize=False,
                                           # betas=content['betas'][frame_idx],
                                           # betas=content['betas'][0],
                                           global_orient=torch.tensor(param['global_orient'][None, frame_idx],
                                                                      device=torch_device),
                                           body_pose=torch.tensor(param['body_pose'][None, frame_idx], device=torch_device),
                                           transl=torch.tensor(param['transl'][None, frame_idx], device=torch_device),
                                           # transl=torch.tensor([param['gt_kp'][frame_idx][0]], device=torch_device),
                                           betas=torch.tensor(param['betas'][None, frame_idx], device=torch_device),
                                           left_hand_pose=torch.tensor(param['left_hand_pose'][None, frame_idx],
                                                                       device=torch_device),
                                           right_hand_pose=torch.tensor(param['right_hand_pose'][None, frame_idx],
                                                                        device=torch_device),
                                           )
                    else:
                        assert False, 'body model not supported'

                    # kp3d = [i[:3] for i in content['keypoints_3d'][frame_idx]]

                # from data_manage.data_vis_demo import draw_gta_skeleton

                # smpl_img = draw_gta_skeleton(smpl_img, frame_idx, seq_p=r'D:\GTAV\visualization\pending\\' + seq)

                img_save_dir_seq = os.path.join(img_save_dir, seq)
                if not os.path.isdir(img_save_dir_seq):
                    os.makedirs(img_save_dir_seq)
                img_save_pathname = os.path.join(img_save_dir_seq, '{:08d}.jpeg'.format(frame_idx))
                cv2.imwrite(img_save_pathname, img)
        except:
            print(seq)

    # # draw pose curves on the distribution
    # fig, axs = plt.subplots(1, 1)
    # # axs.set_title('Pose Sequence')
    # axs.tick_params(axis='x', labelsize=14)
    # axs.tick_params(axis='y', labelsize=14)


def concat_video(overlay_dir, dst, model_type='smpl'):
    # support missing frames video construction

    for seq in os.listdir(overlay_dir):
        if seq.startswith('seq_'):
            pass
        else:
            continue
        img = os.path.join(overlay_dir, seq)

        cmd = (f'ffmpeg.exe -y -r {RECORD_FPS} -f image2 -s 1920x1080 -i {img}\\%8d.jpeg '
               # f'cd {img} && ffmpeg.exe -y -r {RECORD_FPS} -f image2 -s 1920x1080 -i "*.jpeg" '
               f'-vcodec libx264 -crf 25 -pix_fmt yuv420p {dst}\\{seq}_{model_type}.mp4'
               )
        # call(cmd, shell=True)
        os.system(cmd)


if __name__ == '__main__':
    # img_dir_path = r'C:\Users\12595\Desktop\GTA-test-data\mta-for-vis'
    # img_dir_path = r'F:\MTA-multi-p\pending_upload_multi'
    mta_dir = r'E:\gtahuman2_multiple'

    model_mode = 'smpl'
    npz_dir = os.path.join(r'C:\Users\12595\Desktop\GTA-test-data', model_mode, model_mode + '_fitting')
    overlay_output_path = os.path.join(r'C:\Users\12595\Desktop\GTA-test-data', model_mode, model_mode + '_overlay')
    video_output_path = os.path.join(r'C:\Users\12595\Desktop\GTA-test-data', model_mode, model_mode + '_vids')

    specific = 'seq_00087075'
    if model_mode == 'smpl_2d':
        model_mode = 'smplx'
        specific = 'seq_00007522'

    visualize_pose(img_load_dir=mta_dir, npz_load_dir=npz_dir, img_save_dir=overlay_output_path,
                   body_model_type=model_mode, specific_seq=specific, skip_exist=True)
    concat_video(overlay_dir=overlay_output_path, dst=video_output_path, model_type=model_mode)

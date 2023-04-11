import math
import os
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


def visualize_pose(skip_exist=False, img_load_dir='seqs/images', npz_load_dir='seqs/pkls',
                   img_save_dir='seqs/visual', torch_device='cuda', body_model_type='smpl',
                   body_model_path=None, specific_seq=None):
    """Overlays smpl onto images; draw the sequence as curves on the pose distribution"""

    # intrinsic camera
    camera = pyrender.camera.IntrinsicsCamera(fx=1158.0337, fy=1158.0337, cx=960, cy=540)
    kwargs_smpl = {'use_pca': False, 'num_betas': 10,}
    kwargs_smplx = {'flat_hand_mean': True, 'use_face_contour': True, 'use_pca': True, 'num_betas': 10,
                    'num_pca_comps': 24, }

    if body_model_type == 'smpl':
        kwargs = kwargs_smpl
    elif body_model_type == 'smplx':
        kwargs = kwargs_smplx

    body_model = smplx.create(body_model_path,
                              model_type=body_model_type, **kwargs).to(torch_device)

    seqs = sorted(os.listdir(img_load_dir))
    if specific_seq:
        seqs = [specific_seq]

    for seq in tqdm.tqdm(seqs):

        # skip processed seqs
        if skip_exist and os.path.isdir(os.path.join(img_save_dir, seq)):
            continue
        if seq.startswith("seq_") and seq[-3:].isdigit():
            pass
        else:
            continue

        npz_load_pathname = os.path.join(npz_load_dir, seq)
        try:
            param = dict(np.load(npz_load_pathname + '.npz', allow_pickle=True))
        except FileNotFoundError:
            try:
                param = dict(np.load(npz_load_pathname + '.pkl', allow_pickle=True))
            except FileNotFoundError:
                print(npz_load_pathname, 'not found')
                continue

        num_frames = len(os.listdir(os.path.join(img_load_dir, seq)))

        for frame_idx in tqdm.tqdm(range(num_frames)):
            img_load_pathname = os.path.join(img_load_dir, seq, '{:08d}.jpeg'.format(frame_idx))
            img = cv2.imread(img_load_pathname)

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

                                   global_orient=torch.tensor(param['global_orient'][None, frame_idx],
                                                              device=torch_device),
                                   body_pose=torch.tensor(param['body_pose'][None, frame_idx], device=torch_device),
                                   transl=torch.tensor(param['transl'][None, frame_idx], device=torch_device),
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

    model_mode = 'smplx'

    mta_dir = r'E:\gtahuman2_multiple'


    npz_dir = r'C:\Users\12595\Desktop\GTA-test-data\smplx\smplx_fitting'
    # mta_dir = os.path.join(your_path_to_gta_human, 'images')
    overlay_output_path = os.path.join(your_path_to_gta_human, 'overlay')
    video_output_path = os.path.join(your_path_to_gta_human, 'vids')

    # if specific is not none, it only processes 1 specified seq
    specific = None
    # specific = 'seq_00007522'

    # first create image overlay, then make it to videos
    # img_load_dir: gta image path
    # npz_load_dir: the gta smpl fitting, supports both npz and pickle data
    # image_save_dir: output of smpl overlay image
    # dst: video_destination, requires ffmpeg in command line
    visualize_pose(img_load_dir=mta_dir, npz_load_dir=npz_dir, img_save_dir=overlay_output_path,
                   body_model_type=model_mode, body_model_path=r'C:\Users\12595\Desktop\MTA\python\body_models',
                   specific_seq=specific)
    concat_video(overlay_dir=overlay_output_path, dst=video_output_path, model_type=model_mode)

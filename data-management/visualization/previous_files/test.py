import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import smplx
import torch
from tqdm import tqdm
import trimesh

pickle_p = r'D:\GTAV\visualization\pending\annotations'

SMPL_KEYPOINTS = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine_1',
    'left_knee',
    'right_knee',
    'spine_2',
    'left_ankle',
    'right_ankle',
    'spine_3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand',
]

# the full keypoints produced by the default SMPL J_regressor
SMPL_45_KEYPOINTS = SMPL_KEYPOINTS + [
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

GTA_KEYPOINTS = [
    'gta_head_top',  # 00
    'head',  # 01 - head_center
    'neck',  # 02 - neck
    'gta_right_clavicle',  # 03
    'right_shoulder',  # 04  - right_shoulder
    'right_elbow',  # 05  - right_elbow
    'right_wrist',  # 06  - right_wrist
    'gta_left_clavicle',  # 07
    'left_shoulder',  # 08  - left_shoulder
    'left_elbow',  # 09  - left_elbow
    'left_wrist',  # 10  - left_wrist
    'spine_2',  # 11  - spine0
    'gta_spine1',  # 12
    'spine_1',  # 13  - spine2
    'pelvis',  # 14  - pelvis
    'gta_spine4',  # 15
    'right_hip',  # 16  - right_hip
    'right_knee',  # 17  - right_knee
    'right_ankle',  # 18  - right_ankle
    'left_hip',  # 19  - left_hip
    'left_knee',  # 20  - left_knee
    'left_ankle',  # 21  - left_ankle
    'gta_SKEL_ROOT',  # 22
    'gta_FB_R_Brow_Out_000',  # 23
    'left_foot',  # 24  - SKEL_L_Toe0
    'gta_MH_R_Elbow',  # 25
    'left_thumb_2',  # 26  - SKEL_L_Finger01
    'left_thumb_3',  # 27  - SKEL_L_Finger02
    'left_ring_2',  # 28  - SKEL_L_Finger31
    'left_ring_3',  # 29  - SKEL_L_Finger32
    'left_pinky_2',  # 30  - SKEL_L_Finger41
    'left_pinky_3',  # 31  - SKEL_L_Finger42
    'left_index_2',  # 32  - SKEL_L_Finger11
    'left_index_3',  # 33  - SKEL_L_Finger12
    'left_middle_2',  # 34  - SKEL_L_Finger21
    'left_middle_3',  # 35  - SKEL_L_Finger22
    'gta_RB_L_ArmRoll',  # 36
    'gta_IK_R_Hand',  # 37
    'gta_RB_R_ThighRoll',  # 38
    'gta_FB_R_Lip_Corner_000',  # 39
    'gta_SKEL_Pelvis',  # 40
    'gta_IK_Head',  # 41
    'gta_MH_R_Knee',  # 42
    'gta_FB_LowerLipRoot_000',  # 43
    'gta_FB_R_Lip_Top_000',  # 44
    'gta_FB_R_CheekBone_000',  # 45
    'gta_FB_UpperLipRoot_000',  # 46
    'gta_FB_L_Lip_Top_000',  # 47
    'gta_FB_LowerLip_000',  # 48
    'right_foot',  # 49  - SKEL_R_Toe0
    'gta_FB_L_CheekBone_000',  # 50
    'gta_MH_L_Elbow',  # 51
    'gta_RB_L_ThighRoll',  # 52
    'gta_PH_R_Foot',  # 53
    'left_eye',  # 54  - FB_L_Eye_000
    'gta_SKEL_L_Finger00',  # 55
    'left_index_1',  # 56  - SKEL_L_Finger10
    'left_middle_1',  # 57  - SKEL_L_Finger20
    'left_ring_1',  # 58  - SKEL_L_Finger30
    'left_pinky_1',  # 59  - SKEL_L_Finger40
    'right_eye',  # 60  - FB_R_Eye_000
    'gta_PH_R_Hand',  # 61
    'gta_FB_L_Lip_Corner_000',  # 62
    'gta_IK_R_Foot',  # 63
    'gta_RB_Neck_1',  # 64
    'gta_IK_L_Hand',  # 65
    'gta_RB_R_ArmRoll',  # 66
    'gta_FB_Brow_Centre_000',  # 67
    'gta_FB_R_Lid_Upper_000',  # 68
    'gta_RB_R_ForeArmRoll',  # 69
    'gta_FB_L_Lid_Upper_000',  # 70
    'gta_MH_L_Knee',  # 71
    'gta_FB_Jaw_000',  # 72
    'gta_FB_L_Lip_Bot_000',  # 73
    'gta_FB_Tongue_000',  # 74
    'gta_FB_R_Lip_Bot_000',  # 75
    'gta_IK_Root',  # 76
    'gta_PH_L_Foot',  # 77
    'gta_FB_L_Brow_Out_000',  # 78
    'gta_SKEL_R_Finger00',  # 79
    'right_index_1',  # 80  - SKEL_R_Finger10
    'right_middle_1',  # 81  - SKEL_R_Finger20
    'right_ring_1',  # 82  - SKEL_R_Finger30
    'right_pinky_1',  # 83  - SKEL_R_Finger40
    'gta_PH_L_Hand',  # 84
    'gta_RB_L_ForeArmRoll',  # 85
    'gta_FB_UpperLip_000',  # 86
    'right_thumb_2',  # 87  - SKEL_R_Finger01
    'right_thumb_3',  # 88  - SKEL_R_Finger02
    'right_ring_2',  # 89  - SKEL_R_Finger31
    'right_ring_3',  # 90  - SKEL_R_Finger32
    'right_pinky_2',  # 91  - SKEL_R_Finger41
    'right_pinky_3',  # 92  - SKEL_R_Finger42
    'right_index_2',  # 93  - SKEL_R_Finger11
    'right_index_3',  # 94  - SKEL_R_Finger12
    'right_middle_2',  # 95  - SKEL_R_Finger21
    'right_middle_3',  # 96  - SKEL_R_Finger22
    'gta_FACIAL_facialRoot',  # 97
    'gta_IK_L_Foot',  # 98
    'nose'  # 99  - interpolated nose
]


def visualize_pose(skip_exist=False, draw_gta_embedded=False, img_load_dir='seqs/images', pkl_load_dir='seqs/pkls',
                   img_save_dir='seqs/visual'):
    """Overlays smpl onto images; draw the sequence as curves on the pose distribution"""

    camera = pyrender.camera.IntrinsicsCamera(
        fx=1158.0337, fy=1158.0337,
        cx=960, cy=540)

    smpl = smplx.create(r'C:\Users\it_admin\Desktop\mta-2021-1210\python\smpl_visalization',
                        model_type='smpl',
                        gender='neutral',
                        num_betas=10)

    seqs = sorted(os.listdir(img_load_dir))


    wrong_joints_dict = {}

    for seq in tqdm(seqs):

        # skip processed seqs
        if skip_exist and os.path.isdir(os.path.join(img_save_dir, seq)):
            continue
        if seq.startswith("seq_") and seq[-3:].isdigit():
            pass
        else:
            continue

        # load pkl
        pkl_load_pathname = os.path.join(pkl_load_dir, seq + '.pkl')
        with open(pkl_load_pathname, 'rb') as f:
            content = pickle.load(f, encoding='latin1')

        for k in content:
            print(k)

        num_frames = content['num_frames']
        print('frames num =' + str(num_frames))

        wrong_joints = []
        smpl_pts = []

        gta_pts = []

        for frame_idx in tqdm(range(num_frames)):
            img_load_pathname = os.path.join(img_load_dir, seq, '{:08d}.jpeg'.format(frame_idx))
            img = cv2.imread(img_load_pathname)
            kp3d_before_regress = content['kp3d_before_regress'][frame_idx]
            smpl_img, joint3d = draw_overlay(img, smpl, camera, cam_poses=np.eye(4), H=1080, W=1920,
                                             visualize=False,
                                             # betas=content['betas'][frame_idx],
                                             betas=content['betas'][0],
                                             global_orient=content['global_orient'][frame_idx],
                                             body_pose=content['body_pose'][frame_idx],
                                             transl=content['transl'][frame_idx]
                                             )

            kp3d = [i[:3] for i in content['keypoints_3d'][frame_idx]]

            # joint2d = [project_3d_to_2d(pt) for pt in joint3d[0]]
            joint2d = [project_3d_to_2d(pt) for pt in kp3d_before_regress]
            kp2d = [project_3d_to_2d(pt) for pt in kp3d]

            smpl_pts.append(joint2d)
            gta_pts.append(kp2d)


            wrong_c = 0

            for maps in mapping:
                p_smpl, p_gta = maps
                pt1_x, pt1_y = int(joint2d[p_smpl][0]), int(joint2d[p_smpl][1])
                pt2_x, pt2_y = int(kp2d[p_gta][0]), int(kp2d[p_gta][1])
                pt1 = (pt1_x, pt1_y)
                pt2 = (pt2_x, pt2_y)

                dist = abs(pt1_x - pt2_x) + abs(pt1_y - pt2_y)
                print('dist = ' + str(dist))
                if dist >= 15:

                    cv2.line(img, pt1, pt2, (0, 69, 255), thickness=2)
                    cv2.putText(img, SMPL_45_KEYPOINTS[p_smpl] + str(p_smpl), (pt1_x + 100, 300 + wrong_c * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 69, 0), 2, cv2.LINE_AA)
                    cv2.circle(img, pt2, 2, (255, 69, 0), thickness=2)
                    if SMPL_45_KEYPOINTS[p_smpl] not in wrong_joints:
                        wrong_joints.append(SMPL_45_KEYPOINTS[p_smpl])
                    wrong_c += 1
                else:
                    cv2.line(img, pt1, pt2, (255, 255, 255), thickness=2)


            from data_manage.data_vis_demo import draw_gta_skeleton

            # smpl_img = draw_gta_skeleton(smpl_img, frame_idx, seq_p=r'D:\GTAV\visualization\pending\\' + seq)

            img_save_dir_seq = os.path.join(img_save_dir, seq)
            if not os.path.isdir(img_save_dir_seq):
                os.makedirs(img_save_dir_seq)
            img_save_pathname = os.path.join(img_save_dir_seq, '{:08d}.jpeg'.format(frame_idx))
            cv2.imwrite(img_save_pathname, img)



        print(wrong_joints)
        wrong_joints_dict[seq] = wrong_joints
        # find_nearest_relation(mapping, smpl_pts, gta_pts):
    print(wrong_joints_dict)



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
    '''for k, v in kwargs.items():
        print(k, v.shape)'''
    # kwargs['body_pose'] = torch.cat((kwargs['body_pose'][:, :], kwargs['global_orient'][:, :], ), 1)
    model_output = body_model(return_verts=True, **kwargs)

    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    faces = body_model.faces
    joints3d = model_output.joints.detach().cpu().numpy()

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

    body_pose = body_model.body_pose

    return img, joints3d


def create_mapping():

    mapping = []
    for joints in SMPL_45_KEYPOINTS:

        try:
            relation = (SMPL_45_KEYPOINTS.index(joints), GTA_KEYPOINTS.index(joints))
        except ValueError:
            continue

        mapping.append(relation)

    print(mapping)
    return mapping


def project_3d_to_2d(pt, fx=1158.0337, fy=1158.0337,
        cx=960, cy=540):

    [x, y, z] = pt

    u = x * fy / z + cx
    v = y * fy / z + cy

    return [u, v]


def find_nearest_relation(mapping, smpl_pts, gta_pts):

    for relation in mapping:

        smpl_idx, gta_idx = relation






if __name__ == '__main__':

    mapping = create_mapping()
    path = os.path.join(r'D:/GTAV/visualization', 'pending')
    overlay_outpath = r'D:\GTAV\visualization'

    visualize_pose(img_load_dir=path, pkl_load_dir=path + r'\annotations', img_save_dir=overlay_outpath)

    dict1 = {'seq_00000368': ['right_shoulder', 'right_elbow', 'right_wrist', 'right_foot', 'left_wrist', 'neck', 'left_hip'],
     'seq_00000369': ['right_elbow', 'right_wrist', 'right_shoulder', 'neck'],
     'seq_00000370': ['right_wrist', 'left_elbow', 'right_shoulder', 'spine_1', 'left_shoulder', 'left_hip',
                      'left_foot'],
     'seq_00000372': ['right_hip', 'right_elbow', 'right_wrist', 'left_wrist', 'right_ankle', 'right_foot',
                      'left_shoulder', 'left_hip', 'left_ankle'], 'seq_00000375': [],
     'seq_00000376': ['right_elbow', 'right_wrist', 'right_foot', 'left_wrist', 'right_hip', 'right_knee',
                      'right_ankle', 'left_hip', 'left_shoulder'], 'seq_00000377': ['right_wrist'],
     'seq_00000378': ['right_elbow', 'right_wrist'],
     'seq_00000384': ['left_knee', 'right_knee', 'left_ankle', 'left_hip', 'right_shoulder', 'right_wrist',
                      'left_wrist', 'left_foot', 'right_hip', 'right_ankle', 'right_foot', 'left_shoulder',
                      'right_elbow', 'spine_1'],
     'seq_00000389': ['left_knee', 'right_knee', 'left_wrist', 'right_wrist', 'left_ankle', 'left_foot', 'right_ankle',
                      'right_foot', 'right_shoulder', 'neck', 'left_shoulder'],
     'seq_00000390': ['right_elbow', 'right_wrist'],
     'seq_00000402': ['right_ankle', 'spine_1', 'right_knee', 'left_hip'], 'seq_00000403': [],
     'seq_00000404': ['right_knee'], 'seq_00000406': [], 'seq_00000414': ['right_elbow', 'right_wrist'],
     'seq_00000416': ['right_wrist', 'right_knee', 'right_shoulder', 'right_foot'],
     'seq_00000420': ['right_elbow', 'right_wrist'],
     'seq_00000422': ['right_knee', 'right_wrist', 'right_shoulder', 'right_elbow']}

    '''gb = abc['global_orient']
    print(gb[10])
    kp2d = abc['keypoints_2d']
    kp3d = abc['keypoints_3d']
    bp = abc['body_pose']
    # print(bp[10])
    # print(len(kp2d[10]))
    print(kp3d[10])
    tsl = abc['transl']
    
    xywh = abc['bbox_xywh']
    print(len(xywh))
    # 99 original keypoints + 1
    '''


import glob
import math
import os
import os.path as osp
from subprocess import call

import cv2 as cv2
import numpy as np
from tqdm import trange

__all__ = [cv2]

CAM_FOV = 50
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920
RECORD_FPS = 15

src = r"D:\GTAV\MTA"  # path to seq_0000XXXX
dst = r"D:\GTAV\test"  # output

os.makedirs(dst, exist_ok=True)


def draw_pts(img, pts, color=(255, 128, 76)):
    pts = np.float32(pts)
    pts = np.int32(pts)
    cv2.circle(img, tuple(pts), 2, color, -1)


def parse_label(fp):
    data = open(fp, 'r').readlines()
    data = data[1:]
    res_dict = {}
    occ_num = 0
    occ_total = 0
    for line in data:
        items = line.strip().split(',')
        if len(items) < 18:
            break
        (frame, ped_id, ped_action, joint_type, pts_x, pts_y, pts_3D_x,
         pts_3D_y, pts_3D_z, occluded, self_occluded, cam_3D_x, cam_3D_y,
         cam_3D_z, cam_rot_x, cam_rot_y, cam_rot_z, fov, ismale) = items
        res_dict.setdefault(
            int(frame), {
                'peds': {},
                'cam': [
                    cam_3D_x, cam_3D_y, cam_3D_z, cam_rot_x, cam_rot_y,
                    cam_rot_z
                ]
            })
        res_dict[int(frame)]['peds'].setdefault(ped_id, [])
        res_dict[int(frame)]['peds'][ped_id].append([
            pts_x, pts_y, joint_type, ped_action, pts_3D_x, pts_3D_y, pts_3D_z,
            occluded, self_occluded
        ])
        if occluded == '1':
            occ_num += 1
        occ_total += 1
    if occ_num > occ_total * 0.2:
        print(fp, occ_num, occ_total)
    return res_dict


def get_bbox(pts3d, cam_x, cam_y, cam_z, rot_x, rot_y, rot_z):
    # pts3d = pts3d[[0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    pts_a = pts3d.min(0)
    pts_b = pts3d.max(0)
    pts0 = projection(pts_a[0], pts_a[1], pts_a[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts1 = projection(pts_b[0], pts_a[1], pts_a[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts2 = projection(pts_a[0], pts_b[1], pts_a[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts3 = projection(pts_a[0], pts_a[1], pts_b[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts4 = projection(pts_b[0], pts_b[1], pts_a[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts5 = projection(pts_b[0], pts_a[1], pts_b[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts6 = projection(pts_a[0], pts_b[1], pts_b[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts7 = projection(pts_b[0], pts_b[1], pts_b[2], cam_x, cam_y, cam_z, rot_x,
                      rot_y, rot_z)
    pts_2d = np.float32([pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7])
    pts_2d_a = pts_2d.min(0)
    pts_2d_b = pts_2d.max(0)
    return pts_2d_a[0], pts_2d_a[1], pts_2d_b[0], pts_2d_b[1]


def projection(x,
               y,
               z,
               cam_x,
               cam_y,
               cam_z,
               rot_x,
               rot_y,
               rot_z,
               return_depth=False):
    x = x - cam_x
    y = y - cam_y
    z = z - cam_z
    rx = rot_x * math.pi / 180.
    ry = rot_y * math.pi / 180.
    rz = rot_z * math.pi / 180.
    cx = math.cos(rx)
    cy = math.cos(ry)
    cz = math.cos(rz)
    sx = math.sin(rx)
    sy = math.sin(ry)
    sz = math.sin(rz)
    dx = cy * (sz * y + cz * x) - sy * z
    dy = sx * (cy * z + sy * (sz * y + cz * x)) + cx * (cz * y - sz * x)
    dz = cx * (cy * z + sy * (sz * y + cz * x)) - sx * (cz * y - sz * x)
    fov_rad = CAM_FOV * math.pi / 180.0
    f = (SCREEN_HEIGHT / 2.0) * math.cos(fov_rad / 2.0) / math.sin(
        fov_rad / 2.0)
    res_x = (dx * (f / dy)) / SCREEN_WIDTH + 0.5
    res_y = 0.5 - (dz * (f / dy)) / SCREEN_HEIGHT
    res_x *= SCREEN_WIDTH
    res_y *= SCREEN_HEIGHT

    if return_depth:
        return res_x, res_y, dy
    else:
        return res_x, res_y


LIMBS = [
    (0, 1),  # head_top -> head_center
    (1, 2),  # head_center -> neck
    (2, 3),  # neck -> right_clavicle
    (3, 4),  # right_clavicle -> right_shoulder
    (4, 5),  # right_shoulder -> right_elbow
    (5, 6),  # right_elbow -> right_wrist
    (2, 7),  # neck -> left_clavicle
    (7, 8),  # left_clavicle -> left_shoulder
    (8, 9),  # left_shoulder -> left_elbow
    (9, 10),  # left_elbow -> left_wrist
    (2, 11),  # neck -> spine0
    (11, 12),  # spine0 -> spine1
    (12, 13),  # spine1 -> spine2
    (13, 14),  # spine2 -> spine3
    (14, 15),  # spine3 -> spine4
    (15, 16),  # spine4 -> right_hip
    (16, 17),  # right_hip -> right_knee
    (17, 18),  # right_knee -> right_ankle
    (15, 19),  # spine4 -> left_hip
    (19, 20),  # left_hip -> left_knee
    (20, 21),  # left_knee -> left_ankle
]


def check_occ(fs):
    labels = parse_label(fs + '\\peds.csv')


def demo_vis(fs):
    labels = parse_label(fs + '\\peds.csv')

    cmap = np.random.rand(50, 3) * 255
    cmap = np.uint8(cmap)
    ifs = glob.glob(fs + '\\*.jpeg')
    ifs.sort()

    for k in trange(len(ifs)):
        img = cv2.imread(fs + '\\{:08d}.jpeg'.format(k))
        if k in labels:
            for pid in labels[k]['peds']:
                cam = labels[k]['cam']
                cam = np.float32(cam)
                mks = labels[k]['peds'][pid]
                mks3d = [info[4:7] for info in mks]
                mks3d = np.float32(mks3d)
                oc1 = [info[-2] for info in mks]
                oc2 = [info[-1] for info in mks]
                oc1 = np.int32(oc1)
                oc2 = np.int32(oc2)
                _mks = [info[:4] for info in mks]
                acts = [info[-1] for info in _mks]
                mks = [info[:-1] for info in _mks]
                mks = np.float32(mks)
                mks = np.int32(mks)
                bbox = get_bbox(mks3d, *cam)
                depth = projection(*mks3d[2], *cam, return_depth=True)[-1]
                color = tuple(cmap[int(pid) % 50].tolist())

                '''cv2.rectangle(img,
                              tuple(bbox[:2]),
                              tuple(bbox[2:]),
                              color,
                              thickness=2)'''
                for lidx in LIMBS:
                    pi, pj = lidx
                    cv2.line(img,
                             tuple(mks[pi][:2]),
                             tuple(mks[pj][:2]),
                             color=(76, 128, 255),
                             thickness=2)

                for midx, pts in enumerate(mks):
                    action = acts[midx]
                    pts_oc1 = oc1[midx]
                    pts_oc2 = oc2[midx]
                    jtype = str(pts[-1])
                    pts = pts[:2]
                    if pts_oc1 == 1:
                        draw_pts(img, pts, (0, 0, 255))
                    elif pts_oc2 == 1:
                        draw_pts(img, pts, (221, 82, 226))
                    else:
                        draw_pts(img, pts)

                    if jtype == '0':
                        offset = 30
                        x, y = int(pts[0]) - offset, int(pts[1]) - offset
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, pid, (x, y), font, 1,
                        # cv2.putText(img, action, (x, y), font, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(dst + '\\{:06d}.jpg'.format(k), img)

    # concat images to a video
    fn = fs.split("\\")[-1]
    cmd = (
        f'ffmpeg.exe -y -r {RECORD_FPS} -f image2 -s 1920x1080 -i {dst}\\%6d.jpg '
        f'-vcodec libx264 -crf 25 -pix_fmt yuv420p {dst}\\{fn}.mp4'
    )

    call(cmd)

    # remove tmp images
    for fn in glob.glob(dst + '\\*.jpg'):
        os.remove(fn)


def merge_video():
    fs = glob.glob(dst + '\\*.mp4')
    with open(dst + '\\merge_list.txt', 'w') as fo:
        for k in fs:
            fo.write("file '{}'\n".format(k))
    cmd = "ffmpeg.exe  -y -safe 0 -f concat -i {}\\merge_list.txt -c copy {}\\output.mp4".format(
        dst, dst)
    call(cmd, shell=True)


def draw_gta_skeleton(img, frame, seq_p):
    labels = parse_label(seq_p + '\\peds.csv')

    cmap = np.random.rand(50, 3) * 255
    cmap = np.uint8(cmap)
    ifs = glob.glob(seq_p + '\\*.jpeg')
    ifs.sort()

    k = frame - 1

    if k in labels:
        print(111)
        for pid in labels[k]['peds']:
            cam = labels[k]['cam']
            cam = np.float32(cam)
            mks = labels[k]['peds'][pid]
            mks3d = [info[4:7] for info in mks]
            mks3d = np.float32(mks3d)
            oc1 = [info[-2] for info in mks]
            oc2 = [info[-1] for info in mks]
            oc1 = np.int32(oc1)
            oc2 = np.int32(oc2)
            _mks = [info[:4] for info in mks]
            acts = [info[-1] for info in _mks]
            mks = [info[:-1] for info in _mks]
            mks = np.float32(mks)
            mks = np.int32(mks)

            font = cv2.FONT_HERSHEY_SIMPLEX
            x, y = LIMBS[0]
            cv2.putText(img, str(pid), (x, y), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            depth = projection(*mks3d[2], *cam, return_depth=True)[-1]
            color = tuple(cmap[int(pid) % 50].tolist())

            for lidx in LIMBS:
                pi, pj = lidx
                cv2.line(img,
                         tuple(mks[pi][:2]),
                         tuple(mks[pj][:2]),
                         color=(76, 128, 255),
                         thickness=2)

            for midx, pts in enumerate(mks):
                action = acts[midx]
                pts_oc1 = oc1[midx]
                pts_oc2 = oc2[midx]
                jtype = str(pts[-1])
                pts = pts[:2]
                if pts_oc1 == 1:
                    draw_pts(img, pts, (0, 0, 255))
                elif pts_oc2 == 1:
                    draw_pts(img, pts, (221, 82, 226))
                else:
                    draw_pts(img, pts)

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

                offset = 30
                x, y = int(pts[0]) - offset, int(pts[1]) - offset


                if GTA_KEYPOINTS[int(jtype)] in SMPL_KEYPOINTS:
                    pt_name = GTA_KEYPOINTS[int(jtype)]
                    cv2.putText(img, pt_name, (x, y), font, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

    return img


if __name__ == '__main__':

    src = r"C:\Users\12595\Desktop\GTA-test-data\mta-for-vis\seq_00007710"  # path to seq_0000XXXX
    dst = r"C:\Users\12595\Desktop\GTA-test-data\vis-exq"  # output

    if osp.isfile(src + '\\peds.csv'):
        fs_lst = [src]
    else:
        fs_lst = [osp.join(src, fs) for fs in os.listdir(src)]
    for fs in fs_lst:
        try:
            demo_vis(fs)
        except:
            pass

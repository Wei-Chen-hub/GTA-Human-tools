import json
import math
import os
import random
import time
import glob
import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

fov = 50
pi = 3.1415926
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920


def get_camera_extrinsic(log_path):
    with open(log_path) as log_f:
        lines = log_f.readlines()
        for line in lines:
            if line.startswith('Camera pos'):
                camx = float(line.rsplit(' ', 3)[1])
                camy = float(line.rsplit(' ', 3)[2])
                camz = float(line.rsplit(' ', 3)[3])
            if line.startswith('Camera rot'):
                rx = float(line.rsplit(' ', 3)[1])
                ry = float(line.rsplit(' ', 3)[2])
                rz = float(line.rsplit(' ', 3)[3])
                break

    rx = rx * pi / 180
    ry = ry * pi / 180
    rz = rz * pi / 180

    cx = math.cos(rx)
    cy = math.cos(ry)
    cz = math.cos(rz)
    sx = math.sin(rx)
    sy = math.sin(ry)
    sz = math.sin(rz)

    r_matrix = np.mat([[cy * cz, cy * sz, -sy],
                       [cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy],
                       [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy]])
    cam_matrix = np.mat([[camx], [camy], [camz]])
    temp = r_matrix * cam_matrix
    e_matrix = np.mat([[r_matrix[0, 0], r_matrix[0, 1], r_matrix[0, 2], -temp[0, 0]],
                       [r_matrix[1, 0], r_matrix[1, 1], r_matrix[1, 2], -temp[1, 0]],
                       [r_matrix[2, 0], r_matrix[2, 1], r_matrix[2, 2], -temp[2, 0]],
                       [0, 0, 0, 1]])

    return e_matrix


def gen_bbox_lineset(csv, extrinsic):
    # generate bounding box of ped, by extending the key points
    csv.reset_index(drop=True, inplace=True)
    pts3d = csv[['3D_x', '3D_y', '3D_z']].values.tolist()
    # cam_para = csv.iloc[0][11:17]

    '''x, y, z = [], [], []
    for pt in pts3d:
        temp = np.matmul(extrinsic, [pt[0], pt[1], pt[2], 1]).A1
        x.append(temp[0])
        y.append(temp[1] * -1)
        z.append(temp[2])
    xyz_adjust = 0.05
    head_adjust = 0.1

    x1, y1, z1, x2, y2, z2 = min(x) - xyz_adjust, min(y) - xyz_adjust, \
                             min(z) - xyz_adjust, max(x) + xyz_adjust, \
                             max(y) + xyz_adjust, max(z) + head_adjust
    bbox3d_pts = []
    for zz in z1, z2:
        for xx in x1, x2:
            for yy in y1, y2:
                bbox3d_pts.append((xx, yy, zz))'''

    x = [pt[0] for pt in pts3d]
    y = [pt[1] for pt in pts3d]
    z = [pt[2] for pt in pts3d]
    xyz_adjust = 0.1
    head_adjust = 0.15
    foot_adjust = 0.05

    x1, y1, z1, x2, y2, z2 = min(x) - xyz_adjust, min(y) - xyz_adjust, min(z) - foot_adjust, \
                             max(x) + xyz_adjust, max(y) + xyz_adjust, max(z) + head_adjust

    bbox3d_pts = []
    for zz in z1, z2:
        for xx in x1, x2:
            for yy in y1, y2:
                temp = np.matmul(extrinsic, [xx, yy, zz, 1]).A1
                bbox3d_pts.append((temp[0], temp[1], temp[2]))

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    bbox3d_pts = np.array(bbox3d_pts).reshape(8, 3)
    bbox3d_pts[:, 1] = -1 * bbox3d_pts[:, 1]
    lines = [[0, 1], [1, 3], [0, 2], [2, 3], [4, 5], [5, 7],
             [6, 7], [4, 6], [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox3d_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # line_set.transform(extrinsic)

    return line_set


def gen_ptc(img, depth, pinehole_cam, extrinsic):
    # image_d = image_d_raw.reshape(1080, 1920, 1)
    depth_img = np.fromfile(depth, dtype='float32')
    depth_img[depth_img <= 0.001] = 0.001
    depth_img = 1 / (depth_img * 6.666)

    depth_img = o3d.geometry.Image(depth_img.reshape(1080, 1920, 1))
    color_img = o3d.geometry.Image(plt.imread(img))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img,
                                                                  depth_scale=1.0, depth_trunc=10.0,
                                                                  convert_rgb_to_intensity=False)

    ptc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic=pinehole_cam)

    ptc_points = np.asarray(ptc.points)
    ptc_points[:, 1] = -1 * ptc_points[:, 1]
    ptc.points = o3d.utility.Vector3dVector(ptc_points)

    '''points = np.asarray(ptc.points)
    colors = np.asarray(ptc.colors)
    print(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([ptc, axis])'''

    return ptc


def check_pt_in_cuboid(pts, corners):
    """
       cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
       points = array of points with shape (N, 3). N = 1 this case

       Returns the indices of the points array which are outside the cube3d
    """
    b1, b2, b3, b4, t1, t2, t3, t4 = np.asarray(corners)

    dir1 = (t1 - b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2 - b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b3 - b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t4) / 2.0

    # dir_vec = pt - cube3d_center
    cube3d_center_exp = np.expand_dims(cube3d_center, axis=0)
    check_vector_exp = np.expand_dims(np.asarray([dir1, dir2, dir3]), axis=0)
    bbox_size_exp = np.expand_dims(np.asarray([size1, size2, size3]), axis=0)

    dir_vect = pts - cube3d_center_exp

    size = np.absolute(np.einsum('ij,ikj->ik', dir_vect, check_vector_exp))
    # c2 = np.einsum('ij,ik->ij', pts, cube3d_center_expansion)
    mask = size - bbox_size_exp * 0.5 < 0
    mask = np.all(mask, axis=1)

    # print(mask.size)
    '''res1 = np.absolute(np.dot(dir_vec, dir1)) * 2 < size1
    res2 = np.absolute(np.dot(dir_vec, dir2)) * 2 < size2
    res3 = np.absolute(np.dot(dir_vec, dir3)) * 2 < size3'''
    # print(res1, res2, res3)
    # if list(set().union(res1, res2, res3))[0] >= 1:
    return mask


def get_partial_ped_ptc(ptc_dict, bbox_dict):
    partial_ptc_dict = {}
    keys_for_delete = []
    for key_ptc in ptc_dict.keys():
        masks, mask_names = [], []
        ptc = ptc_dict[key_ptc]
        ptc_points = np.asarray(ptc.points)
        ptc_colors = np.asarray(ptc.colors)
        for key_bbox in sorted(bbox_dict.keys()):

            if int(key_ptc[4:8]) == int(key_bbox[-8:-4]):
                bbox_pts = np.asarray(bbox_dict[key_bbox].points)
                mask = check_pt_in_cuboid(ptc_points, bbox_pts)
                if True not in mask:
                    keys_for_delete.append(key_bbox)
                masks.append(mask)
                mask_names.append(key_bbox[5:11])

        for idx, mask in enumerate(masks):
            ptc_partial = o3d.geometry.PointCloud()

            ptc_partial.points = o3d.utility.Vector3dVector(ptc_points[mask])

            # indices = np.random.permutation(len(ptc_points[mask]))[:2048] # [:int(len(ptc_points[mask]) / 10)]
            # ptc_partial = ptc_partial.select_by_index(indices)

            # ptc_partial = ptc_partial.farthest_point_down_sample(2048)
            # print(ptc_partial)

            partial_ptc_dict[f'pcd_{mask_names[idx]}_{key_ptc[-8:-4]}.pcd'] = ptc_partial

        mask_all = np.logical_or.reduce(np.array(masks))

        ptc_background = o3d.geometry.PointCloud()
        ptc_background.points = o3d.utility.Vector3dVector(ptc_points[~mask_all])

        partial_ptc_dict[f'bg_{key_ptc[4:8]}.pcd'] = ptc_background

    for key in keys_for_delete:
        print(f'{key} need to delete')
        del bbox_dict[key]

    return partial_ptc_dict, bbox_dict


def process_sequence_ptc(mta_path, seq, dst=None, specific_frame=None, generate='11ps'):
    assert '1' in generate, 'No item generated'
    assert generate[2] in ['p', 'f'], 'Please select full(f) ptc or ped_ptc(p) in generate[2]'
    assert generate[3] in ['s', 'v'], 'Please select save(s) ptc or return obj for vis(v) in generate[3]'
    assert not (generate[3] == 's' and dst is None), 'For save mode in generate[3], dst cannot be None'

    log_p = os.path.join(mta_path, seq, 'log.txt')
    extrinsic = get_camera_extrinsic(log_path=log_p)

    intrinsic = np.array([[1.15803376e+03, 0.00000000e+00, SCREEN_WIDTH / 2],
                          [0.00000000e+00, -1.15803376e+03, SCREEN_HEIGHT / 2],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    cam_param = o3d.camera.PinholeCameraIntrinsic()
    cam_param.set_intrinsics(int(intrinsic[0, 2]) * 2, int(intrinsic[1, 2]) * 2,
                       intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])

    ptc_dict, bbox_dict = {}, {}
    if generate[0] == '1':
        if specific_frame:
            img_p = os.path.join(mta_path, seq, '{:08d}.jpeg'.format(specific_frame))
            depth_p = os.path.join(mta_path, seq, 'raws', 'depth_{:08d}.raw'.format(specific_frame))
            ptc_dict['pcd_{:04d}.pcd'.format(specific_frame)] = gen_ptc(img=img_p, depth=depth_p, pinehole_cam=cam_param, extrinsic=extrinsic)
        else:
            for file in os.listdir(os.path.join(mta_path, seq)):
                if file.endswith('.jpeg'):
                    frame = int(file[:-5])
                    img_p = os.path.join(mta_path, seq, file)
                    depth_p = os.path.join(mta_path, seq, 'raws', 'depth_{:08d}.raw'.format(frame))
                    ptc_dict['pcd_{:04d}.pcd'.format(frame)] = gen_ptc(img=img_p, depth=depth_p, pinehole_cam=cam_param, extrinsic=extrinsic)

    if generate[1] == '1':
        csv_p = os.path.join(mta_path, seq, 'peds.csv')
        ped_df = pd.read_csv(csv_p)
        ped_id_list = ped_df['ped_id'].unique()
        if specific_frame:
            ped_frame_df = ped_df[ped_df['frame'] == specific_frame]
            for ped_id in ped_id_list:
                ped_id_df = ped_frame_df[ped_frame_df['ped_id'] == ped_id]
                bbox_dict['bbox_{:06d}_{:04d}.ply'.format(ped_id, specific_frame)] \
                    = gen_bbox_lineset(ped_id_df, extrinsic)
        else:
            for file in os.listdir(os.path.join(mta_path, seq)):
                if file.endswith('.jpeg'):
                    frame = int(file[:-5])
                    ped_frame_df = ped_df[ped_df['frame'] == frame]
                    for ped_id in ped_id_list:
                        ped_id_df = ped_frame_df[ped_frame_df['ped_id'] == ped_id]
                        bbox_dict['bbox_{:06d}_{:04d}.ply'.format(ped_id, frame)] \
                            = gen_bbox_lineset(ped_id_df, extrinsic)

    if generate[2] == 'p':
        ptc_dict, bbox_dict = get_partial_ped_ptc(ptc_dict, bbox_dict)

    if generate[3] == 'v':
        if specific_frame:
            return list(ptc_dict.values()) + list(bbox_dict.values())
        else:
            return ptc_dict, bbox_dict

    if specific_frame:
        # noinspection PyTypeChecker
        for key, value in ptc_dict.items():
            o3d.io.write_point_cloud(os.path.join(dst, key), value)
        # noinspection PyTypeChecker
        for key, value in bbox_dict.items():
            o3d.io.write_line_set(os.path.join(dst, key), value)
    else:
        os.makedirs(os.path.join(dst, seq), exist_ok=True)
        # noinspection PyTypeChecker
        for key, value in ptc_dict.items():
            o3d.io.write_point_cloud(os.path.join(dst, seq, key), value)
            '''ptc_points = np.asarray(file.points)
            ptc_colors = np.asarray(file.colors)
            ptc_points = np.asarray([[x[0], x[1] * -1, x[2]] for x in ptc_points])
            np.save(os.path.join(dst, seq, name + '_points.npy'), ptc_points)
            np.save(os.path.join(dst, seq, name + '_colors.npy'), ptc_colors)'''
        # noinspection PyTypeChecker
        for key, value in bbox_dict.items():
            o3d.io.write_line_set(os.path.join(dst, seq, key), value)


if __name__ == '__main__':
    mta_path = r'E:\gtahuman2_multiple'
    seq = 'seq_00007522'

    # process_sequence_ptc(mta_path, seq, specific_frame=10, generate='11fv')

    seqs = glob.glob(os.path.join(mta_path, 'seq*'))

    for idx, seq in enumerate(tqdm.tqdm(seqs)):
        if idx < 0:
            continue
        log_path = os.path.join(seq, 'log.txt')
        try:
            process_sequence_ptc(mta_path, os.path.basename(seq), dst=r'E:\gtahuman2_multiple_ptcbbox')  # specific_frame=10,
            # print(seq, 'processed at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        except Exception as e:
            print(seq, 'fails:', e)


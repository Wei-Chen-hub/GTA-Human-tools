import atexit
import json
import math
import os
import random
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

# import time

fov = 50
pi = 3.1415926
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

warnings.filterwarnings("ignore", category=DeprecationWarning)


def down_sample_depth_image(image, ds_scale=2):
    width_c, height_c = 1920, 1080

    if width_c % ds_scale != 0 or height_c % ds_scale != 0:
        raise ValueError('Down-sampled depth map must have integer ds_scale')

    width_d, height_d = int(width_c / ds_scale), int(height_c / ds_scale)

    x_d, y_d = np.meshgrid(np.linspace(0, 1, height_d), np.linspace(0, 1, width_d))
    x_c, y_c = np.meshgrid(np.linspace(0, 1, height_c), np.linspace(0, 1, width_c))

    print(x_d.shape)
    print(y_d.shape)

    dx, dy = 959, 539
    cx, cy = int(x_d[dx][dy] * width_c) - 1, int(y_d[dx][dy] * height_c) - 1
    print(cx, cy)

    dx, dy = int(x_c[cx][cy] * width_d) - 1, int(y_c[cx][cy] * height_d) - 1
    print(dx, dy)

    image_ds = cv2.resize(image, (image.shape[1] // ds_scale,
                                  image.shape[0] // ds_scale),
                          interpolation=cv2.INTER_AREA)

    return image_ds


def cameraM_cal(rx, ry, rz, camx, camy, camz):
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


def generate_ptc_from_depth(depth_image, cx, cy, fx, fy, scalingFactor=1.0):
    rows = depth_image.shape[0]
    cols = depth_image.shape[1]

    points = np.zeros((rows * cols, 3))

    for v in range(rows):
        for u in range(cols):
            # print(depth_image[v, u])
            if depth_image[v, u] == 0:
                Z = 1000
            else:
                Z = scalingFactor / depth_image[v, u]

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            # print(Z)
            if Z == 0:
                continue
            try:
                X, Y, Z = X / 6.66, Y * -1 / 6.66, Z / 6.66
                xyz = [X, Y, Z]
                points[v * cols + u] = xyz
            except ValueError:
                print('xyz = ', X, Y, Z)

    return points


def get_color(img, rows, cols):
    img_row = img.shape[0]
    img_col = img.shape[1]
    channels = img.shape[2]
    np_colors = np.zeros((rows * cols, channels))
    for v in range(rows):
        for u in range(cols):
            np_colors[v * cols + u] = img[int((v / rows) * img_row), int((u / cols) * img_col)] / 256.

    return np_colors


# def projection2d_3d(x, y, cam_x, cam_y, cam_z, rot_x, rot_y, rot_z):
def cal_distance(x, y, z, cam_para):
    [cam_x, cam_y, cam_z, rot_x, rot_y, rot_z] = cam_para
    return math.sqrt((x - cam_x) ** 2 + (y - cam_y) ** 2 + (z - cam_z) ** 2)


def projection3d_2d(x, y, z, cam_para, return_depth=False):
    [cam_x, cam_y, cam_z, rot_x, rot_y, rot_z] = cam_para
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
    fov_rad = fov * math.pi / 180.0
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


def get_bbox(pts3d, cam_para):
    # pts3d = pts3d[[0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    pts_a = pts3d.min(0)
    pts_b = pts3d.max(0)
    pts0 = projection3d_2d(pts_a[0], pts_a[1], pts_a[2], cam_para)
    pts1 = projection3d_2d(pts_b[0], pts_a[1], pts_a[2], cam_para)
    pts2 = projection3d_2d(pts_a[0], pts_b[1], pts_a[2], cam_para)
    pts3 = projection3d_2d(pts_a[0], pts_a[1], pts_b[2], cam_para)
    pts4 = projection3d_2d(pts_b[0], pts_b[1], pts_a[2], cam_para)
    pts5 = projection3d_2d(pts_b[0], pts_a[1], pts_b[2], cam_para)
    pts6 = projection3d_2d(pts_a[0], pts_b[1], pts_b[2], cam_para)
    pts7 = projection3d_2d(pts_b[0], pts_b[1], pts_b[2], cam_para)
    pts_2d = np.float32([pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7])
    pts_2d_a = pts_2d.min(0) - 2
    pts_2d_b = pts_2d.max(0) + 2

    xyz_adjust = 0.1
    head_adjust = 0.15

    bbox_2d = int(pts_2d_a[0]), int(pts_2d_a[1]), int(pts_2d_b[0]), int(pts_2d_b[1])
    bbox_3d = pts3d.min(0)[0] - xyz_adjust, pts3d.min(0)[1] - xyz_adjust, pts3d.min(0)[2] - xyz_adjust, pts3d.max(0)[
        0] + xyz_adjust, \
              pts3d.max(0)[1] + xyz_adjust, pts3d.max(0)[2] + head_adjust

    return bbox_2d, bbox_3d


def gen_mask_2d(csv, extrinsic, pts):
    mask_ds_2d = np.full((1080 * 1920), False)

    csv.reset_index(drop=True, inplace=True)
    pts3d = csv[['3D_x', '3D_y', '3D_z']]
    cam_para = csv.iloc[0][11:17]
    bbox2d, bbox3d = get_bbox(pts3d, cam_para)
    x1, y1, x2, y2 = bbox2d
    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1

    check_error = False
    if box_width >= 1920 - x1 or box_width >= 1920:
        check_error = True
    if box_height >= 1080 - y1 or box_height >= 1080:
        check_error = True
    if check_error:
        coordinates = [(x, y) for x in range(1920) for y in range(1080)]
    else:
        coordinates = [(x + x1, y + y1) for x in range(box_width) for y in range(box_height)]

    '''for coordinate in coordinates:
        (x, y) = coordinate
        mask_ds_2d[y * 1920 + x] = True'''

    x1, y1, z1, x2, y2, z2 = bbox3d
    bb_pts = []
    for z in z1, z2:
        for x in x1, x2:
            for y in y1, y2:
                # dist = cal_distance(x, y, z, cam_para)
                # bbox3d_ptx, bbox3d_pty = projection3d_2d(x, y, z, cam_para)
                # pts.append((bbox3d_ptx, bbox3d_pty, dist))

                temp = np.matmul(extrinsic, [x, y, z, 1]).A1
                # print(temp[0], temp[1], temp[2])
                bb_pts.append((temp[0], -1 * temp[1], temp[2]))

    random.shuffle(coordinates)

    for pt in coordinates:
        try:
            [x, y, z] = pts[pt[1] * 1920 + pt[0]]
        except IndexError:
            continue
        # if z >50: print(x, y, z)
        inside = check_pt_in_cuboid(np.array([x, y, z]), bb_pts)
        if inside:
            mask_ds_2d[pt[1] * 1920 + pt[0]] = True
            # print('Points remained: ', np.count_nonzero(mask))
        '''if np.count_nonzero(mask_ds_2d) >= 2048:
            break'''

    return mask_ds_2d, coordinates, bb_pts


def check_pt_in_cuboid(pt, corners):
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

    dir_vec = pt - cube3d_center
    # print('size=', size1, size2, size3)
    # print('length=', np.dot(dir_vec, dir1), np.dot(dir_vec, dir2), np.dot(dir_vec, dir3))
    # print('length=', np.dot(dir2, dir1), np.dot(dir3, dir2), np.dot(dir1, dir3))

    '''res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]'''
    res1 = np.absolute(np.dot(dir_vec, dir1)) * 2 < size1
    res2 = np.absolute(np.dot(dir_vec, dir2)) * 2 < size2
    res3 = np.absolute(np.dot(dir_vec, dir3)) * 2 < size3
    # print(res1, res2, res3)
    # if list(set().union(res1, res2, res3))[0] >= 1:
    if res1 and res2 and res3:
        return True
    else:
        return False


def draw_bbox(bbox3d_pts, extrinsic):
    # To return
    bbox3d_pts = np.array(bbox3d_pts).reshape(8, 3)
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 3], [0, 2], [2, 3],
             [4, 5], [5, 7], [6, 7], [4, 6],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox3d_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # line_set.transform(extrinsic)

    return line_set


def generate_pt_cloud(image_c, image_d, log_path, ped_id_df, down_scale=None, visible_length=8, mask_input=False,
                      visualize=False):
    if not log_path:
        raise ValueError('Log file needed')
    elif os.path.isfile(log_path):
        # print('Camera data comes from: ' + log_path)
        pass
    if not down_scale:
        down_scale = 1

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

    extrinsic = cameraM_cal(rx, ry, rz, camx, camy, camz)
    # print(extrinsic)
    cam = o3d.camera.PinholeCameraIntrinsic()
    intrinsic = np.array([[1.15803376e+03, 0.00000000e+00, 9.60000000e+02],
                          [0.00000000e+00, -1.15803376e+03, 5.40000000e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    fx = intrinsic[0, 0] / down_scale
    fy = intrinsic[1, 1] / down_scale
    cx = intrinsic[0, 2] / down_scale
    cy = intrinsic[1, 2] / down_scale

    rows = int(1080 / down_scale)
    cols = int(1920 / down_scale)
    cam.set_intrinsics(cols, rows, fx, fy, cx, cy)

    xyz = generate_ptc_from_depth(image_d, cx, cy, fx, fy)
    # print(xyz)

    mask_depth = image_d > 0.1 / visible_length
    mask_depth = mask_depth.reshape(rows * cols)
    mask, coordinates, bbox3d_pts = gen_mask_2d(ped_id_df, extrinsic, xyz)
    if not mask_input:
        mask = mask_depth
    bbox = draw_bbox(bbox3d_pts, extrinsic)
    # mask = np.logical_and(mask_depth, mask_input)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[mask])

    # color = color[:, :, [2, 1, 0]]  # Convert to BGR
    # color = np.fromfile(color_path, dtype='uint8')
    color = image_c.reshape(1080, 1920, 3)
    # color = color[:, :, :3]
    # points = np.asarray(pcd.points)
    np_colors = get_color(color, rows, cols)

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    # pcd.transform(extrinsic)
    # points = np.asarray(pcd.points)
    # print('bbox3d points: ', bbox3d_pts)

    pcd.points = o3d.utility.Vector3dVector(xyz[mask])
    pcd.colors = o3d.utility.Vector3dVector(np_colors[mask])
    # points = np.asarray(pcd.points)
    # pcd.transform(extrinsic)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # pcd = pcd.select_by_index(np.where(mask == True))
    if visualize:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(pcd)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # vis.add_geometry([pcd])
        if bbox3d_pts:
            o3d.visualization.draw_geometries([pcd, bbox])
        else:
            # vis.visualization.draw_geometries([pcd])
            o3d.visualization.draw_geometries([pcd])
        # time.sleep(10000)
        # vis.destroy_window()
    else:
        return pcd, bbox


def gen_seq_ptc(mta_p, seq, dest=None, specific_frame=None, exclude_environment=True, in_original_folder=False):
    for image in os.listdir(os.path.join(mta_p, seq)):
        if not str(image).endswith('.jpeg'):
            continue
        filename = str(image).rsplit('.', 1)[0]
        frame = int(filename)
        if frame < 3:
            continue
        if specific_frame:
            if not frame == specific_frame:
                continue
        color_path = os.path.join(mta_p, seq, image)
        depth_path = os.path.join(mta_p, seq, 'raws', 'depth_' + str(filename) + '.raw')
        stencil_path = os.path.join(mta_p, seq, 'raws', 'stencil_' + str(filename) + '.raw')
        summary_path = os.path.join(mta_p, seq, 'summary.txt')
        log_path = os.path.join(mta_p, seq, 'log.txt')
        csv_path = os.path.join(mta_p, seq, 'peds.csv')
        with open(summary_path, 'r') as f:
            ped_id_list = list(f.readlines()[0][1:-1].split(', '))
        ped_df = pd.read_csv(csv_path)
        ped_df = ped_df[ped_df['frame'] >= 3]

        # frame_num = 5
        frame_num = int(filename)
        image_c = plt.imread(color_path)
        image_d_raw = np.fromfile(depth_path, dtype='float32')
        image_d = image_d_raw.reshape(1080, 1920, 1)
        if not in_original_folder:
            os.makedirs(os.path.join(dest, seq), exist_ok=True)
        pcd = []
        bbox = []
        for ped_id in ped_id_list:
            # print(ped_id)
            ped_id_df = ped_df[ped_df['ped_id'] == int(ped_id)]
            ped_id_df = ped_id_df[ped_id_df['frame'] == int(frame_num)]
            # print(seq, ped_id)

            pcd_ped, bbox_ped = generate_pt_cloud(image_c, image_d, log_path, ped_id_df, down_scale=None,
                                                  visible_length=8,
                                                  mask_input=exclude_environment,
                                                  visualize=False)

            pcd.append(pcd_ped)
            bbox.append(bbox_ped)
            if in_original_folder:
                pcd_filename = os.path.join(mta_p, seq, 'smpl_ptc', ped_id + '_pcd_' + filename + '.pcd')
                bbox_filename = os.path.join(mta_p, seq, 'smpl_ptc', ped_id + '_bbox_' + filename + '.ply')
            else:
                pcd_filename = os.path.join(dest, seq, ped_id + '_pcd_' + filename + '.pcd')
                bbox_filename = os.path.join(dest, seq, ped_id + '_bbox_' + filename + '.ply')
            o3d.io.write_point_cloud(pcd_filename, pcd_ped)
            o3d.io.write_line_set(bbox_filename, bbox_ped)
            print('Processed sequence: ', seq, 'frame: ', frame_num, 'ped ID: ', ped_id, 'at ',
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # look_at = np.average(look_at, 0).tolist()

        # print('look at: ', look_at)
        '''o3d.visualization.draw_geometries(pcd + bbox,
                                          front=[1, 0, 0],
                                          lookat=look_at,
                                          up=[0, 1, 0],
                                          zoom=1)'''


def request_for_seqs(mta_path, log_path, seq_num=1):
    with open(log_path, 'r') as f:

        logs = json.load(f)

    seqs_generated = [seq for seq in logs if logs[seq] == 'success'] + [seq for seq in logs if logs[seq] == 'requested']
    # print(seqs_generated)

    seqs_all = os.listdir(mta_path)
    seqs_all.sort()

    #
    seqs = [seq for seq in seqs_all if seq not in seqs_generated]
    seqs.sort()
    # print(seqs)
    seqs = seqs[:seq_num]

    '''del_list = []
    for k, v in logs.items():
        if logs[k] == 'requested':
            del_list.append(k)
    for k in del_list:
        del logs[k]
'''

    update = {}
    for seq in seqs:
        update[seq] = 'requested'
    with open(log_path, 'r') as f:
        logs = json.load(f)
    for seq in update:
        logs[seq] = update[seq]
    with open(log_path, 'w') as f:
        json.dump(logs, f)
    # seqs = [seq for seq in seqs if seq not in failed_seqs]
    # print('seqs pending: ', len(seqs))

    return seqs, logs


def exit_handler(sth, logs, log_path):
    print('failed seqs: ', sth)
    for seq in sth:
        logs[seq] = 'failed'
    with open(log_path, 'w') as f:
        json.dump(logs, f)


def process_some_sequence():
    log_path = os.path.join(r'\\DESKTOP-LV0VM5K\pc2_c\Users\it_admin\Desktop\GTA-test-data\ptc_bbox', 'log_ptc.json')
    mta_path = r'\\DESKTOP-LV0VM5K\gta_multi_ssd1\MTA-multi-p\pending_upload_multi'
    # seqs = ['seq_00007496', 'seq_00007740', 'seq_00009114']
    # seqs = ['seq_00009114']
    image = '00000007.jpeg'
    dest = r'\\DESKTOP-LV0VM5K\pc2_c\Users\it_admin\Desktop\GTA-test-data\ptc_bbox'

    failed_seqs = ['seq_00009356']

    seq_num = 1
    seqs, logs = request_for_seqs(mta_path=mta_path, log_path=log_path, seq_num=seq_num)
    update = {}
    for index in range(seq_num):
        try:
            print(seqs[index])
            gen_seq_ptc(mta_path, seqs[index], dest)
            update[seqs[index]] = 'success'
        except IndexError:
            failed_seqs.append(seqs[index])
            failed_seqs = list(set(failed_seqs))
            atexit.register(exit_handler, sth=failed_seqs, logs=logs, log_path=log_path)
        pass

    with open(log_path, 'r') as f:
        logs = json.load(f)
    for seq in update:
        logs[seq] = update[seq]
    with open(log_path, 'w') as f:
        json.dump(logs, f)


def updater():
    dest = r'\\DESKTOP-LV0VM5K\pc2_c\Users\it_admin\Desktop\GTA-test-data\ptc_bbox'
    log_path = os.path.join(r'\\DESKTOP-LV0VM5K\pc2_c\Users\it_admin\Desktop\GTA-test-data\ptc_bbox', 'log_ptc.json')

    seqs = os.listdir(dest)

    update = {}

    with open(log_path, 'r') as f:
        logs = json.load(f)
    for seq in seqs:
        if seq in logs.keys():
            pass
        else:
            update[seq] = 'success'
    for seq in update:
        logs[seq] = update[seq]
    with open(log_path, 'w') as f:
        json.dump(logs, f)

    print('Processed seqs updated')


if __name__ == '__main__':

    while True:
        updater()
        # process_some_sequence()

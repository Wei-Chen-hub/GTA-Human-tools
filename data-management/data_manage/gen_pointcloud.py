import math
import os

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920
fov = 50
pi = 3.1415926


def raw_view_gta(img_path):
    temp, filename = os.path.split(img_path)
    filename = filename.rsplit('.', 1)[0]
    depth_path = os.path.join(temp, 'raws', 'depth_' + filename + '.raw')
    stencil_path = os.path.join(temp, 'raws', 'stencil_' + filename + '.raw')
    print(depth_path)

    rows = 1080
    cols = 1920
    channels = 4
    img_c = plt.imread(img_path)
    plt.imshow(img_c)
    plt.show()

    print('Cannot find a corresponding color image!')

    if os.path.isfile(depth_path):
        rows = 1080
        cols = 1920
        channels = 1

        img_d = np.fromfile(depth_path, dtype='float32')
        print(img_d)
        try:
            img_d = img_d.reshape(rows, cols, channels)
        except ValueError:
            rows = 720
            cols = 1280
            channels = 1
            img_d = img_d.reshape(rows, cols, channels)
        plt.imshow(img_d)
        plt.show()
    else:
        print('Cannot find a corresponding depth image!')

    if os.path.isfile(stencil_path):
        rows = 1080
        cols = 1920
        channels = 1

        img_s = np.fromfile(stencil_path, dtype='uint8')
        try:
            img_s = img_s.reshape(rows, cols, channels)
        except ValueError:
            rows = 720
            cols = 1280
            channels = 1
            img_s = img_s.reshape(rows, cols, channels)
        print(np.unique(img_s))
        print(img_s[500, 900])

        plt.imshow(img_s)
        plt.show()
    else:
        print('Cannot find a corresponding stencil image!')


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
    channels = depth_image.shape[2]
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
            points[v * cols + u] = [X, Y, Z]

    return points


def get_color(img, rows, cols):
    img_row = img.shape[0]
    img_col = img.shape[1]
    channels = img.shape[2]
    np_colors = np.zeros((rows * cols, 3))
    for v in range(rows):
        for u in range(cols):
            np_colors[v * cols + u] = img[int((v / rows) * img_row), int((u / cols) * img_col)] / 256.

    return np_colors


def generate_pt_cloud(img_path, log_path=None, visualize=False, visible_length=8):
    temp, filename = os.path.split(img_path)
    if not log_path:
        log_path = os.path.join(temp, 'log.txt')
        if os.path.isfile(log_path):
            print('Camera data comes from: ' + log_path)
        else:
            print('Cannot find a log data')
            return 0
    filename = filename.rsplit('.', 1)[0]
    depth_path = os.path.join(temp, 'raws', 'depth_' + filename + '.raw')
    # color_path = os.path.join(temp, 'raws', 'color_' + filename + '.raw')
    color_path = img_path
    if not os.path.isfile(depth_path):
        print('Cannot find a corresponding depth image!')

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
    print(extrinsic)
    cam = o3d.camera.PinholeCameraIntrinsic()
    intrinsic = np.array([[1.15803376e+03, 0.00000000e+00, 9.60000000e+02],
                          [0.00000000e+00, -1.15803376e+03, 5.40000000e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    channels = 1
    rows = 1080
    cols = 1920
    cam.set_intrinsics(cols, rows, fx, fy, cx, cy)

    depth = np.fromfile(depth_path, dtype='float32')
    # depth = cv2.imread(depth_path)
    mask_depth = depth > 0.1 / visible_length

    try:
        depth = depth.reshape(rows, cols, channels)
    except ValueError:
        rows = 720
        cols = 1280
        cam.set_intrinsics(cols, rows, fx, fy, cx, cy)
        depth = depth.reshape(rows, cols, channels)
    # print(depth)

    xyz = generate_ptc_from_depth(depth, cx, cy, fx, fy)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    color = cv2.imread(color_path)
    color = color[:, :, [2, 1, 0]]  # Convert to BGR
    # color = np.fromfile(color_path, dtype='uint8')
    color = color.reshape(1080, 1920, 3)
    # color = color[:, :, :3]

    np_colors = get_color(color, rows, cols)

    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    pcd.transform(extrinsic)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(points)
    print(mask_depth)
    # pcd = pcd.select_by_index(np.where(mask == True))
    if visualize:
        # print(pcd)
        pcd.points = o3d.utility.Vector3dVector(points[mask_depth])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask_depth])
        o3d.visualization.draw_geometries([pcd])
    else:
        return pcd, points, colors, mask_depth


if __name__ == '__main__':
    # for files in os.listdir()
    imgp = r'F:\MTA-multi-p\pending_upload_multi\seq_00007675\00000007.jpeg'
    # imgp = r'D:\GTAV\MTA\seq_00023893\00000011.jpeg'
    raw_view_gta(imgp)  # 可以查看某张图片对应的depth map
    # generate_pt_cloud(imgp, visible_length=5)  # default visible_length = 10

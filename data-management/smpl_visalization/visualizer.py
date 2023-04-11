import argparse

import cv2
import numpy as np
import open3d as o3d
import pyrender
import smplx
import torch
import trimesh

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
res = 224

# smpl visualization
W, H = 1920, 1080
fx = fy = 1000
cx, cy = 960, 540
male_body_model = smplx.create(r'C:\Users\it_admin\Desktop\mta-2021-1210\python\smpl_visalization',
                               model_type='smpl',
                               gender='male',
                               num_betas=10)
female_body_model = smplx.create(r'C:\Users\it_admin\Desktop\mta-2021-1210\python\smpl_visalization',
                                 model_type='smpl',
                                 gender='female',
                                 num_betas=10)
neutral_body_model = smplx.create(r'C:\Users\it_admin\Desktop\mta-2021-1210\python\smpl_visalization',
                                  model_type='smpl',
                                  gender='neutral',
                                  num_betas=10)
cam_poses = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, -5],
    [0, 0, 0, 1]
])

print(cam_poses)


def preprocess_img(img):
    # dimension
    img = img.transpose(1, 2, 0)
    # color
    img = img[:, :, ::-1]
    # denormalize
    img = img * std + mean
    # clamp
    img[img < 0.0] = 0.0
    img[img > 1.0] = 1.0
    # dtype
    img = img * 255
    img = img.astype(np.uint8)
    return img


def draw_2d_keypoints(img, keypoints):
    keypoints = (keypoints + 1) * res / 2.0
    for kp in keypoints:  # dimension
        x, y, conf = kp
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255) if conf > 0.5 else (255, 0, 0), thickness=-1)
    return img


def draw_3d_keypoints(keypoints3d):
    pc = keypoints3d[:, :-1]
    conf = keypoints3d[:, -1]

    color = np.zeros((len(pc), 3))
    color[conf > 0.5] = np.array([1.0, 0.0, 0.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)  # numpy (n, 3)
    pcd.colors = o3d.utility.Vector3dVector(color)  # numpy (n, 3), 0-1, optional
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    visual = [pcd, axis]
    o3d.visualization.draw_geometries(visual)


def draw_3d_keypoints_correpondences(gt, pd):
    visual = []

    # draw points
    num_points = len(gt)
    pd_color = np.array([1.0, 0.0, 0.0]).reshape(1, 3).repeat(num_points, axis=0)
    gt_color = np.array([0.0, 1.0, 0.0]).reshape(1, 3).repeat(num_points, axis=0)

    points = np.vstack([gt, pd])
    point_colors = np.vstack([gt_color, pd_color])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    visual.append(pcd)

    # draw correspondences
    idx = [(i, i + num_points) for i in range(num_points)]
    line_colors = np.array([0.0, 0.0, 1.0]).reshape(1, 3).repeat(num_points, axis=0)

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(idx))
    lineset.colors = o3d.utility.Vector3dVector(line_colors)

    # visualize
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    visual.append(axis)
    o3d.visualization.draw_geometries(visual)


def draw_smpl(pose, shape, gender):
    if gender == 0:
        body_model = male_body_model
    elif gender == 1:
        body_model = female_body_model
    else:
        body_model = neutral_body_model

    model_output = body_model(return_verts=True,
                              betas=torch.Tensor(np.expand_dims(shape, axis=0)),
                              global_orient=torch.Tensor(np.expand_dims(pose[:3], axis=0)),
                              body_pose=torch.Tensor(np.expand_dims(pose[3:], axis=0)),
                              # betas=torch.zeros((1, 10)),
                              # global_orient=torch.Tensor([[0, 0, 0]]),
                              # body_pose=torch.zeros((1, 69)),
                              # transl=torch.Tensor([[0,0,0]])
                              )
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
    camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera, pose=cam_poses)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                             viewport_size=(W, H),
                             cull_faces=False,
                             run_in_thread=False,
                             registered_keys=dict())


def visualize_input(content):
    num_frames = len(content['dataset_name'])
    for i in range(num_frames):

        dataset_name = content['dataset_name'][i]
        imgname = content['imgname'][i].split('/')[-2]
        if dataset_name == 'gta-smplx':
            print(i, '/', num_frames, dataset_name, imgname)
        else:
            print(i, '/', num_frames, dataset_name)

        img = preprocess_img(content['img'][i])

        keypoints = content['keypoints'][i]
        img_2d_keypoints = draw_2d_keypoints(img.copy(), keypoints)

        cv2.imshow('2d_keypoints', img_2d_keypoints)
        cv2.waitKey(0)

        has_pose_3d = content['has_pose_3d'][i]
        if has_pose_3d:
            keypoints3d = content['pose_3d'][i]
            draw_3d_keypoints(keypoints3d)

        has_smpl = content['has_smpl'][i]
        if has_smpl:
            pose = content['pose'][i]
            shape = content['betas'][i]
            gender = content['gender'][i]
            draw_smpl(pose, shape, gender)


def visualize_eval(content):
    num_frames = len(content['gender'])
    for i in range(num_frames):
        img = preprocess_img(content['img'][i])
        cv2.imshow('img', img)
        cv2.waitKey(0)

        keypoints3d_gt = content['gt_joints'][i]
        keypoints3d_pd = content['pred_joints'][i]
        draw_3d_keypoints_correpondences(keypoints3d_gt, keypoints3d_pd)

        import pdb;
        pdb.set_trace()

        pose = content['pose'][i]
        shape = content['betas'][i]
        gender = content['gender'][i]
        draw_smpl(pose, shape, gender)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'eval'], help='file is saved from train or eval')
    parser.add_argument('filename', type=str, help='filename')
    args = parser.parse_args()

    content = dict(np.load(args.filename))
    print(content.keys())

    if args.mode == 'eval':
        visualize_eval(content)
    else:
        visualize_input(content)

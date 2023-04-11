import os
import time

import numpy as np
import open3d as o3d
import smplx
import torch
import trimesh
from subprocess import call



def concat_video(overlay_dir, dst):
    for seq in os.listdir(overlay_dir):
        if seq.startswith('seq_'):
            pass
        else:
            continue
        img = os.path.join(overlay_dir, seq, 'vis')

        '''cmd = (f'ffmpeg.exe -y -r {RECORD_FPS} -f image2 -s 1920x1061 -i {img}\\%8d.png '
               f'-vcodec libx264 -crf 25 -pix_fmt yuv420p {dst}\\{seq}_ptc.mp4')'''
        cmd = f"ffmpeg -r 10 -i {img}\\%8d.png -vcodec mpeg4  -vf scale=-1:1080 -y {dst}\\{seq}_ptc.mp4"

        call(cmd, shell=True)


def gen_smpl(param_path, idx):
    body_model = smplx.create(r'visalization',
                              model_type='smpl', gender='neutral', num_betas=10)

    torch_device = torch.device('cpu')
    param = dict(np.load(param_path))

    output = body_model(
        global_orient=torch.tensor(param['global_orient'][None, idx], device=torch_device),
        body_pose=torch.tensor(param['body_pose'][None, idx], device=torch_device),
        transl=torch.tensor(param['transl'][None, idx], device=torch_device),
        betas=torch.tensor(param['betas'][None, idx], device=torch_device),
        return_verts=True
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    # print(vertices)

    trimesh_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
    body_model_mesh = trimesh_mesh.as_open3d
    body_model_mesh.compute_vertex_normals()

    return body_model_mesh


def process():
    process_seq_num = 200
    processed = 0
    for seq in seqs:
        processed += 1
        if not seq.startswith('seq'):
            continue
        images = os.listdir(os.path.join(dest, seq))
        images = [x for x in images if x[0].isdigit()]
        ped_id_list = []
        frame_id_list = []

        for image in images:
            ped_id = image.rsplit('_', 2)[0]
            ped_id_list.append(int(ped_id))
            frame_id = int(image.rsplit('_', 3)[2].rsplit('.', 2)[0])
            frame_id_list.append(frame_id)
            frame_id_list.sort()

        ped_id_list = list(set(ped_id_list))
        frame_id_list = list(set(frame_id_list))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, visible=False)

        parameters = o3d.io.read_pinhole_camera_parameters(
            r'C:\Users\12595\Desktop\MTA\python\ScreenCamera3.json')
        ctr = vis.get_view_control()

        video = []

        for frame in frame_id_list:

            filename = '{:08d}'.format(frame)
            pcd = []
            bbox = []
            look_at = []
            body_mesh = []

            for ped_id in ped_id_list:
                pcd_filename = os.path.join(dest, seq, str(ped_id) + '_pcd_' + filename + '.pcd')
                bbox_filename = os.path.join(dest, seq, str(ped_id) + '_bbox_' + filename + '.ply')

                pcd_ped = o3d.io.read_point_cloud(pcd_filename)
                bbox_ped = o3d.io.read_line_set(bbox_filename)
                look_at_ped = np.average(np.asarray(bbox_ped.points), 0)

                points = np.asarray(pcd_ped.points)
                points = points / 6.66
                for index in range(len(points)):
                    points[index][1] = points[index][1] * -1
                pcd_ped.points = o3d.utility.Vector3dVector(points)

                '''points = np.asarray(bbox_ped.points)
                points = points / 6.66
                print(points)
                for index in range(len(points)):
                    points[index][1] = points[index][1] * -1
                bbox_ped.points = o3d.utility.Vector3dVector(points)'''

                use_npy = True
                if use_npy:
                    try:
                        pc_path = r'C:\Users\it_admin\Desktop\GTA-test-data\gta_human++\point_cloud_1019'
                        pc_path = os.path.join(pc_path, seq + '_' + str(ped_id) + '.npy')
                        point_cloud = np.load(pc_path)
                        # pcd_ped = o3d.geometry.PointCloud()
                        pcd_ped.points = o3d.utility.Vector3dVector(point_cloud[frame - 3])
                    except:
                        pass

                pcd.append(pcd_ped)
                bbox.append(bbox_ped)
                look_at.append(look_at_ped)

                try:
                    param_path = r'C:\Users\it_admin\Desktop\GTA-test-data\gta_human++\param_1019'
                    smpl_filename = seq + '_' + str(ped_id) + '.npz'
                    body_model_mesh = gen_smpl(os.path.join(param_path, smpl_filename), idx=frame - 3)
                    body_mesh.append(body_model_mesh)
                except Exception as e:
                    print(e)
                    pass

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # geometries = pcd + bbox + body_mesh + [axis]
            geometries = pcd + body_mesh + [axis]

            '''look_at = np.average(look_at, 0)
            front = [-look_at[1], -look_at[0], -look_at[2]]
            up = [-look_at[0], -look_at[1], look_at[2]]
            # print('look at ', look_at)
            o3d.visualization.draw_geometries(geometries, width=1920, height=1080, left=400, top=300)'''

            for mygeometry in geometries:
                vis.add_geometry(mygeometry)
                vis.update_geometry(mygeometry)
            ctr.convert_from_pinhole_camera_parameters(parameters)
            vis.poll_events()
            vis.update_renderer()
            # option 1: use capture_screen_image

            # option 2: use capture_screen_float_buffer

            time.sleep(0.3)
            os.makedirs(os.path.join(dest, seq, 'vis'), exist_ok=True)

            video.append(np.array(vis.capture_screen_float_buffer()))
            vis.capture_screen_image(os.path.join(dest, seq, 'vis', f'{int(frame):08d}.png'))

            for mygeometry in geometries:
                vis.remove_geometry(mygeometry)
            print('processed: ', seq, ' frame: ', frame, ' at ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        vis.destroy_window()

        del vis
        del ctr

        if processed >= process_seq_num:
            break

if __name__ == '__main__':
    mta_p = r'C:\Users\12595\Desktop\GTA-test-data\mta-for-vis'
    # seqs = ['seq_00007496', 'seq_00007740', 'seq_00009114']

    seqs = ['seq_00009114']
    image = '00000007.jpeg'
    # dest = r'C:\Users\it_admin\Desktop\GTA-test-data\ptc_bbox'
    # seqs = os.listdir(dest)

    RECORD_FPS = 15


    overlay_out_path = r'C:\Users\12595\Desktop\GTA-test-data\smplx-overlay'
    video_destination = r'C:\Users\it_admin\Desktop\GTA-test-data\vids'
    # concat_video(overlay_dir=overlay_out_path, dst=video_destination)

    process()

'''# print(1)
    video_save_path = os.path.join(dest, 'videos', seq + '.mp4')
    fps = 15
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
    # out = cv2.VideoWriter(video_save_path, 'MJPG', fps, (1920, 1080))
    for frame in video:
        # print(frame.shape)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)'''

'''camera = o3d.camera.PinholeCameraIntrinsic()
down_scale = 1
width, height = 1920, 1080
intrinsic = np.array([[1.15803376e+03, 0.00000000e+00, 9.60000000e+02],
                      [0.00000000e+00, -1.15803376e+03, 5.40000000e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
fx = intrinsic[0, 0] / down_scale
fy = intrinsic[1, 1] / down_scale
cx = intrinsic[0, 2] / down_scale
cy = intrinsic[1, 2] / down_scale

rows = int(1080 / down_scale)
cols = int(1920 / down_scale)
camera_params = o3d.camera.PinholeCameraParameters()
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width, height, fx, fy, cx, cy)
camera_params.intrinsic = intrinsic
camera_params.extrinsic = np.eye(4)'''

import json
import os
import pickle
import sys

import numpy as np
import open3d as o3d
import pandas as pd
import tqdm


def check_smpl(seq_path, smpl_path, require_all=True):
    summary_path = os.path.join(seq_path, 'summary.txt')
    with open(summary_path, 'r') as f:
        ped_id_list = list(f.readlines()[0][1:-1].split(', '))

    mta_path, seq = os.path.split(seq_path)
    generated_list = []
    for ped_id in ped_id_list:
        filename = seq + '_' + str(ped_id) + '.pkl'
        if os.path.exists(os.path.join(smpl_path, filename)):
            generated_list.append(ped_id)

    if len(generated_list) == len(ped_id_list):
        return True, generated_list
    elif len(generated_list) == 0:
        return False, generated_list
    else:
        if require_all:
            return False, generated_list
        else:
            return True, generated_list


def check_ptc_bbox(seq_path, ptc_path, require_all=True):
    summary_path = os.path.join(seq_path, 'summary.txt')
    csv_path = os.path.join(seq_path, 'peds.csv')
    with open(summary_path, 'r') as f:
        ped_id_list = list(f.readlines()[0][1:-1].split(', '))
    ped_df = pd.read_csv(csv_path)
    frame_count = ped_df['frame'].max() - 3

    mta_path, seq = os.path.split(seq_path)
    generated_list = []
    for ped_id in ped_id_list:
        all_exist = True
        for frame in range(3, frame_count + 3):

            filename1 = str(ped_id) + '_bbox_' + '{:08d}'.format(frame) + '.ply'
            filename2 = str(ped_id) + '_pcd_' + '{:08d}'.format(frame) + '.pcd'
            if not os.path.exists(os.path.join(ptc_path, seq, filename1)):
                all_exist = False
                break
            if not os.path.exists(os.path.join(ptc_path, seq, filename2)):
                all_exist = False
                break
        if all_exist:
            generated_list.append(ped_id)

    if len(generated_list) == len(ped_id_list):
        return True, generated_list
    elif len(generated_list) == 0:
        return False, generated_list
    else:
        if require_all:
            return False, generated_list
        else:
            return True, generated_list


def convert_ptc(ptc_dir, seq, ped_id_list, output_dir, down_sample=2048, save_bbox=False, save_color=False):
    ptc_dir = str(os.path.join(ptc_dir, seq))
    ptc_files_all = os.listdir(ptc_dir)
    for ped_id in ped_id_list:
        ptc_list, ptc_list_color, bbox_list = np.empty((0, down_sample, 3)), np.empty((0, down_sample, 3)), np.array([])
        ptc_files_ped = [f for f in ptc_files_all if f.startswith(str(ped_id)) and f.endswith('pcd')]
        if save_bbox:
            bbox_files_ped = [f for f in ptc_files_all if f.startswith(str(ped_id)) and f.endswith('ply')]

        for ptc_file in ptc_files_ped:
            ptc_ped = o3d.io.read_point_cloud(os.path.join(ptc_dir, ptc_file))
            # print(np.asarray(ptc_ped.points).shape)
            ptc_ped_points = np.asarray(ptc_ped.points)

            if save_color:
                ptc_ped_colors = np.asarray(ptc_ped.colors)

            for index in range(len(ptc_ped_points)):
                ptc_ped_points[index][1] = ptc_ped_points[index][1] * -1

            if down_sample:
                np.random.shuffle(ptc_ped_points)
                # c = np.c_[ptc_ped_points, ptc_ped_colors]
                # np.random.shuffle(c)
                # ptc_ped_points, ptc_ped_colors = c[:, :ptc_ped_points.shape[1]], c[:, :ptc_ped_colors.shape[1]]
                temp = np.expand_dims(ptc_ped_points[:down_sample], 0)
                # temp_colors = np.expand_dims(ptc_ped_colors[:down_sample], 0)
            else:
                temp = np.expand_dims(ptc_ped_points, 0)
                # temp_colors = np.expand_dims(ptc_ped_colors, 0)

            ptc_list = np.concatenate((ptc_list, temp))
            # ptc_list_color = np.concatenate((ptc_list_color, temp_colors))
        ptc_filename = seq + '_' + '{:06d}'.format(int(ped_id))
        output_file = os.path.join(output_dir, ptc_filename)

        # scale_factor = 2000 * 0.15 / (0.15 + 50 * (2000-0.15))
        scale_factor = 6.66
        ptc_list = ptc_list / scale_factor
        # ptc_list = ptc_list[2:]
        # ptc_list_color = ptc_list_color[2:]
        np.save(output_file, ptc_list)
        # np.save(os.path.join(output_dir, seq + '_' + str(ped_id) + '_color'), ptc_list_color)

        # print(ptc_list[1])
        # print(ptc_list_color[1])


def convert_smpl(smpl_path, seq, ped_id_list, output_dir):
    for ped_id in ped_id_list:
        smpl_filename = seq + '_' + str(ped_id) + '.pkl'

        with open(os.path.join(smpl_path, smpl_filename), 'rb') as f:
            smpl_data = pickle.load(f, encoding='latin1')
        npz_dict = {k: v[3:] for k, v in smpl_data.items() if k in ['global_orient', 'body_pose', 'betas', 'transl']}
        npz_name = smpl_filename[:12] + '_' + '{:06d}'.format(int(smpl_filename[13:-4]))

        output_file = os.path.join(output_dir, npz_name)
        np.savez(output_file, **npz_dict)


def clear_data(dir_npy, dir_npz):
    for files in os.listdir(dir_npz):
        if not os.path.exists(os.path.join(dir_npy, files[:-3]) + 'npy'):
            print(files[:-3], 'npy : does not exist, remove corresponding npz !!!')
            os.remove(os.path.join(dir_npz, files))

    for files in os.listdir(dir_npy):
        if not os.path.exists(os.path.join(dir_npz, files[:-3]) + 'npz'):
            print(files[:-3], 'npz : does not exist, remove corresponding npy !!!')
            os.remove(os.path.join(dir_npy, files))


def update_log(seq, status, which_log, log_path=None):

    if not log_path:
        if which_log == 'smpl':
            log_path = r'C:\Users\12595\Desktop\GTA-test-data\smpl_fitting\log_smpl.json'
        if which_log == 'ptc':
            log_path = r'D:\ptc\log_ptc.json'

    with open(log_path, 'r') as f:
        logs = json.load(f)

    try:
        if status:
            logs[seq] = 'success'
        else:
            if logs[seq] == 'success':
                del logs[seq]
                print(which_log, 'of', seq, 'not really generated, revert log back')
            elif logs[seq] == 'requested':
                del logs[seq]
                print(which_log, 'of', seq, 'is requested but not really generated, revert log back')
        with open(log_path, 'w') as f:
            json.dump(logs, f)

    except KeyError as e:
        # print(e)
        pass

    # print(logs)


if __name__ == '__main__':

    soruce_mta = r'F:\MTA-multi-p\pending_upload_multi'
    source_ptc = r'D:\ptc'
    source_smpl = r'C:\Users\12595\Desktop\GTA-test-data\smpl_fitting'
    dest_final = r'C:\Users\12595\Desktop\GTA-test-data\gta-human++'
    dir_npy = os.path.join(dest_final, 'point_cloud')
    dir_npz = os.path.join(dest_final, 'param')

    '''update_log(seq=1, which_log='smpl')
    sys.exit(0)'''
    # clear_data(dir_npy=dir_npy, dir_npz=dir_npz)

    files = os.listdir(soruce_mta)
    # files = files[360:]
    # for folder in files:
    #    update_log(folder, status='success', which_log='smpl')

    # for folder in tqdm.tqdm(files, position=0, leave=True):
    for folder in files:
        try:
            # print('Processing: ', folder)
            status_smpl, processed_smpl = check_smpl(seq_path=os.path.join(soruce_mta, folder),
                                                     smpl_path=source_smpl)
            status_ptc, processed_ptc = check_ptc_bbox(seq_path=os.path.join(soruce_mta, folder),
                                                       ptc_path=source_ptc)
            if status_smpl:
                convert_smpl(smpl_path=source_smpl, seq=folder, ped_id_list=processed_smpl, output_dir=dir_npz)
                # print('Convert SMPL: ', folder, ': Success')
                # print(len(processed_smpl))
            if status_ptc:
                convert_ptc(ptc_dir=source_ptc, seq=folder, ped_id_list=processed_ptc, output_dir=dir_npy)
                # print('Convert PTC: ', folder, ': Success')
                pass

            update_log(folder, status=status_smpl, which_log='smpl')
            update_log(folder, status=status_ptc, which_log='ptc')

        except ValueError:
            # print('ValueError in', folder, 'skip')
            continue
        except OSError:
            print('OSError in', folder, 'skip')
            continue
    clear_data(dir_npy=dir_npy, dir_npz=dir_npz)


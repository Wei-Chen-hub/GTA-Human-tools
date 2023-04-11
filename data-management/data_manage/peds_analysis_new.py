import os
import pandas as pd
import numpy as np
import math

# from utils import move_folder

''' 
if ped_csv_tag is -1: 
    person walk out of the screen, or person is not loaded
else:
    2 anim_time too long
    3 ped is somehow occluded, either by object or self
    9 mostly occluded
    4 occlusion is caused by self occlusion
    5 ped not moving
    6 ped moves but mainly drifting
    7 ped is likely not a normal standing pose (crouch or lay down pose)
    8 ped is doing a non-physical action (e.g. sitting in the air)
    10 ped shows inconsistent location
    11 camera under ground
'''

walk_out_percentage = 0.6  # above this, person not in screen
occluded_heavy = 0.6
occluded_light = 0.1
self_occlusion_threshold = 0.5  # to classify self or object occlusion
lowest_moving_speed = 0.01  # to check if not moving
not_moving_threshold = 0.5  # percentage
lowest_moving_accel = 0.005  # to check if drifting
drifting_threshold = 0.5  # percentage
joint_num = 99
spine_foot_height_threshold = 0.5  # difference between spine and foot average
sampling_rate = 0.08  # video sampling
valid_physics_threshold = 0.05  # to check for non-physical action
valid_physics_threshold_percentage = 0.1  # percentage
max_moving_threshold = 2  # check for inconsistent location
long_anim_threshold = 300  # more than this frame number is not ideal / current 150 limit in generation


def analysis_peds_csv(ped_csv, ped_csv_tag=1):
    ped_df = pd.read_csv(ped_csv)
    ped_df_all = ped_df[ped_df['frame'] >= 3]
    id_list = pd.unique(ped_df['ped_id'])

    # Separate different ped into ped[n]
    ped_count = len(id_list)
    ped = {}
    ped_csv_tags_dict = {}

    for name in id_list:
        ped_df = ped_df_all[ped_df_all['ped_id'] == name]
        frame_list = pd.unique(ped_df['frame'])
        ped_csv_tag = 1

        # check if camera under ground
        check_frame = 5
        camera_z = ped_df[ped_df['frame'] == check_frame]['cam_3D_z'].mean()

        '''        if ped_df[ped_df['frame'] == check_frame]['3D_z'].dtype() != np.number:
            # check for nan values
            ped_csv_tag = -1
            ped_csv_tags_dict[name] = ped_csv_tag
            return ped_csv_tags_dict'''

        try:
            person_lowset = ped_df[ped_df['frame'] == check_frame]['3D_z'].mean()
        except TypeError:
            ped_csv_tag = -1
            ped_csv_tags_dict[name] = ped_csv_tag
            return ped_csv_tags_dict

        if camera_z < person_lowset - 0.2:
            ped_csv_tag *= 23

        # analysis for ped walk out of the screen
        walk_out_frame = 0
        for frame in frame_list:
            x_2d_mean = ped_df[ped_df['frame'] == frame]['2D_x'].mean()
            y_2d_mean = ped_df[ped_df['frame'] == frame]['2D_y'].mean()
            if 0 < float(x_2d_mean) < 1920 and 0 < float(y_2d_mean) < 1080:
                pass
            else:
                walk_out_frame += 1
        if walk_out_frame >= len(frame_list) * walk_out_percentage:
            ped_csv_tag *= -1

        # analysis for occlusion
        occluded_frame_num = 0
        for frame in frame_list:
            if ped_df[ped_df['frame'] == frame]['occluded'].mean() >= 0.95:
                occluded_frame_num += 1
        if occluded_frame_num >= 0.25 * len(frame_list):
            ped_csv_tag *= 9
            # return ped_csv_tag

        occlusion = ped_df['occluded'].mean()
        self_occlusion = ped_df['self_occluded'].mean()
        if occlusion >= occluded_heavy:
            ped_csv_tag *= 9
            if occlusion - self_occlusion < self_occlusion_threshold * occlusion:
                ped_csv_tag *= 5
            else:
                # ped_csv_tag *= -1
                pass

        elif occluded_heavy > occlusion >= occluded_light:
            ped_csv_tag *= 3
            if occlusion - self_occlusion <= self_occlusion_threshold * occlusion:
                ped_csv_tag *= 5
        else:
            pass

        # analysis for ped not moving
        joint_list = pd.unique(ped_df['joint_type'])
        ped_joint_moving = {}
        ped_3d_location = ped_df[['frame', 'ped_id', 'joint_type', '3D_x', '3D_y', '3D_z']]
        not_moving_joint = 0
        drifting_joint = 0
        inconsistent_locale = 0

        for joint in joint_list:
            ped_joint_moving[joint] = ped_3d_location[ped_3d_location['joint_type'] == joint].astype(float)

            # calculate speed
            ped_joint_moving[joint] = ped_joint_moving[joint].diff(axis=0)
            speed_x = ped_joint_moving[joint]['3D_x'].apply(abs).median()
            speed_y = ped_joint_moving[joint]['3D_y'].apply(abs).median()
            speed_z = ped_joint_moving[joint]['3D_z'].apply(abs).median()
            max_speed_x = ped_joint_moving[joint]['3D_x'].apply(abs).max()
            max_speed_y = ped_joint_moving[joint]['3D_y'].apply(abs).max()
            max_speed_z = ped_joint_moving[joint]['3D_z'].apply(abs).max()

            # calculate acceleration
            ped_joint_moving[joint] = ped_joint_moving[joint].diff(axis=0)
            accel_x = ped_joint_moving[joint]['3D_x'].apply(abs).median()
            accel_y = ped_joint_moving[joint]['3D_y'].apply(abs).median()
            accel_z = ped_joint_moving[joint]['3D_z'].apply(abs).median()

            if speed_x + speed_y + speed_z < lowest_moving_speed:
                not_moving_joint += 1
            if accel_x + accel_y + accel_z < lowest_moving_accel:
                drifting_joint += 1
            if max_speed_x + max_speed_y + max_speed_z > max_moving_threshold:
                inconsistent_locale += 1

        if not_moving_joint >= 99 * not_moving_threshold:
            ped_csv_tag *= 7
        elif drifting_joint >= 99 * drifting_threshold:
            ped_csv_tag *= 9
        if inconsistent_locale >= 4:
            ped_csv_tag *= 19

        # ped is likely not a normal standing pose
        foot_joint_set = [18, 21, 24, 49, 53, 63, 77, 98]
        spine_joint_set = [11, 12, 13, 14, 15]
        max_frame = int(frame_list[-1])
        float_frame_count = 0

        #  Get 5 sample is enough
        for frame in [3, int(max_frame * 0.25), int(max_frame * 0.4), int(max_frame * 0.55), int(max_frame * 0.7)]:
            ped_foot_spine = ped_3d_location[ped_3d_location['frame'] == frame]

            foot_joint_height = 0
            for joint in foot_joint_set:
                temp = ped_foot_spine[ped_foot_spine['joint_type'] == joint]
                foot_joint_height += float(temp['3D_z'])
            foot_joint_height_average = foot_joint_height / 8

            spine_joint_height = 0
            for joint in spine_joint_set:
                temp = ped_foot_spine[ped_foot_spine['joint_type'] == joint]
                spine_joint_height += float(temp['3D_z'])
            spine_joint_height_average = spine_joint_height / 5

            spine_foot_height_difference = spine_joint_height_average - foot_joint_height_average

            if spine_foot_height_difference <= spine_foot_height_threshold:
                float_frame_count += 1

        if float_frame_count >= 3:
            ped_csv_tag *= 13

        # check if ped is doing a non-physical action (e.g. sitting in the air)
        count = 0
        invalid_count = 0
        frame = 3

        while True:
            ped_foot_spine = ped_3d_location[ped_3d_location['frame'] == frame]
            foot_joint_x = []
            foot_joint_y = []
            for joint in foot_joint_set:
                temp = ped_foot_spine[ped_foot_spine['joint_type'] == joint]
                foot_joint_x.append(float(temp['3D_x']))
                foot_joint_y.append(float(temp['3D_y']))
            foot_joint_x_min = min(foot_joint_x)
            foot_joint_x_max = max(foot_joint_x)
            foot_joint_y_min = min(foot_joint_y)
            foot_joint_y_max = max(foot_joint_y)

            spine_joint_x = 0
            spine_joint_y = 0
            for joint in spine_joint_set:
                temp = ped_foot_spine[ped_foot_spine['joint_type'] == joint]
                spine_joint_x += float(temp['3D_x'])
                spine_joint_y += float(temp['3D_y'])
            spine_joint_x = spine_joint_x / 5
            spine_joint_y = spine_joint_y / 5

            x_distance = min(abs(spine_joint_x - foot_joint_x_min), abs(spine_joint_x - foot_joint_x_max))
            y_distance = min(abs(spine_joint_y - foot_joint_y_min), abs(spine_joint_y - foot_joint_y_max))
            if foot_joint_x_min <= spine_joint_x <= foot_joint_x_max:
                if foot_joint_y_min <= spine_joint_y <= foot_joint_y_max:
                    physics_distance = 0
                else:
                    physics_distance = y_distance
            else:
                if foot_joint_y_min <= spine_joint_y <= foot_joint_y_max:
                    physics_distance = x_distance
                else:
                    physics_distance = math.sqrt(x_distance * x_distance + y_distance * y_distance)
            if physics_distance >= valid_physics_threshold:
                invalid_count += 1
            count += 1

            frame += int(sampling_rate * max_frame)
            if frame > max_frame:
                break

        if invalid_count >= count * valid_physics_threshold_percentage:
            ped_csv_tag *= 17

        if max_frame >= long_anim_threshold:
            ped_csv_tag *= 2

        ped_csv_tags_dict[name] = ped_csv_tag

    return ped_csv_tags_dict


def get_seperate_tags(tags_mul):
    real_tags_dict = {}

    for key in tags_mul:
        real_tags_list = []
        key_tags = tags_mul[key]

        if key_tags <= 0:
            real_tags_list.append(-1)
        if key_tags % 2 == 0:
            real_tags_list.append(2)
        if key_tags % 9 == 0:
            real_tags_list.append(9)
        elif key_tags % 3 == 0:
            real_tags_list.append(3)
        if key_tags % 5 == 0:
            real_tags_list.append(4)
        if key_tags % 7 == 0:
            real_tags_list.append(5)
        if key_tags % 11 == 0:
            real_tags_list.append(6)
        if key_tags % 13 == 0:
            real_tags_list.append(7)
        if key_tags % 17 == 0:
            real_tags_list.append(8)
        if key_tags % 19 == 0:
            real_tags_list.append(10)
        if key_tags % 23 == 0:
            real_tags_list.append(11)
        if not real_tags_list:
            real_tags_list.append(1)
        real_tags_list.sort()
        real_tags_dict[key] = real_tags_list

    return real_tags_dict


def get_seq_name(sequence_path):
    sequence_name = sequence_path.split('/')[-1].split('_')[-1]
    return sequence_name


def dynamo_create_item(table, data):
    # utility for create a new entry in dynamo
    # Item={'ID': "Seq-%8d-frame", "batch":"2", "tag":'1'}
    print("Enqueueing: {}".format(data))
    table.put_item(Item=data)


def post_processing(seq_path, destination):
    seq_name = get_seq_name(seq_path)
    ped_csv_path = os.path.join(seq_path, 'peds.csv')
    tags_multiplication_list = analysis_peds_csv(ped_csv_path, ped_csv_tag=1)
    real_seperate_tags_list = get_seperate_tags(tags_multiplication_list)
    real_seperate_tags_list = sorted(real_seperate_tags_list)
    print("real_seperate_tags_list:", real_seperate_tags_list)
    # move_folder(seq_path, destination)
    return os.path.basename(seq_path), " ".join([str(i) for i in real_seperate_tags_list])


if __name__ == '__main__':
    for d in os.listdir(r'D:\GTAV\MTA'):
        input_sequence_path = os.path.join(r'D:\GTAV\MTA', d)
        # post_processing(input_sequence_path)
        # print(post_processing(input_sequence_path))
        ped_csv_path = os.path.join(input_sequence_path, 'peds.csv')

        print(d, get_seperate_tags(analysis_peds_csv(ped_csv_path, ped_csv_tag=1)))


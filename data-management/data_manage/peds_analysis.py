import os
import pandas as pd
import math

#  ped_csv_tag = 1

''' ped_csv_tag if negative, video should be in trash bin,else if can be divided by:
2 anim_time too long
3 ped is somehow occluded, either by object or self
    9 mostly occluded
5 occlusion is caused by self occlusion
7 ped not moving
11 ped moves but mainly drifting
13 ped is likely not a normal standing pose (crouch or  lay down pose)
17 ped is doing a non-physical action (e.g. sitting in the air)
19 ped shows inconsistent location
'''

''' Parameters'''
occluded_heavy = 0.6
occluded_light = 0.1
self_occlusion_threshold = 0.5  # to classify self or object occlusion
occluded_threshold = 0.6  # percentage
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
long_anim_threshold = 300  # more than this frame number is not ideal


def analysis_peds_csv(ped_csv, ped_csv_tag=1):
    ped_df = pd.read_csv(ped_csv)
    ped_df = ped_df[ped_df['frame'] >= 3]
    # print(ped_df.dtypes)
    # print(pd.unique(ped_df['ped_id']))

    id_list = pd.unique(ped_df['ped_id'])

    # Separate different ped into ped[n]
    count = 0
    ped = {}
    for name in id_list:
        ped[count] = ped_df[ped_df['ped_id'] == name]
        count += 1

    foot_joint_set = []
    hand_joint_set = []
    spine_joint_set = []
    head_joint_set = []

    # analysis for occlusion
    frame_list = pd.unique(ped_df['frame'])
    max_frame = int(frame_list[-1])
    occluded_frame_num = 0
    for frame in frame_list:
        if ped_df[ped_df['frame'] == frame]['occluded'].mean() >= 0.95:
            occluded_frame_num += 1
    if occluded_frame_num >= 0.25 * len(frame_list):
        ped_csv_tag *= -1
        return ped_csv_tag

    light_occluded_frame_num = 0
    heavy_occluded_frame_num = 0
    self_occluded_frame_num = 0

    for frame in frame_list:
        frame_occlusion = ped_df[ped_df['frame'] == frame]['occluded'].mean()
        frame_self_occlusion = ped_df[ped_df['frame'] == frame]['self_occluded'].mean()
        if frame_occlusion >= occluded_heavy:
            heavy_occluded_frame_num += 1
            if frame_occlusion - frame_self_occlusion <= self_occlusion_threshold * frame_occlusion:
                self_occluded_frame_num += 1
        elif occluded_heavy > frame_occlusion >= occluded_light:
            light_occluded_frame_num += 1

    if heavy_occluded_frame_num >= occluded_threshold * max_frame:
        ped_csv_tag *= 9
        if self_occluded_frame_num >= occluded_threshold * max_frame:
            ped_csv_tag *= 5
    elif light_occluded_frame_num >= occluded_threshold * max_frame:
        ped_csv_tag *= 3
        if self_occluded_frame_num >= occluded_threshold * max_frame:
            ped_csv_tag *= 5
    else:
        pass

    # print(ped[0]['occluded'].mean())
    # print(ped[0]['self_occluded'].mean())
    # print(ped_csv_tag)

    # analysis for ped not moving
    joint_list = pd.unique(ped_df['joint_type'])
    # print(joint_list)
    ped_joint_moving = {}
    ped_3d_location = ped[0][['frame', 'ped_id', 'joint_type', '3D_x', '3D_y', '3D_z']]
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
    return ped_csv_tag


# analysis_peds_csv(ped_csv_path)
def extract_free_fall(ped_csv, freefall_location='', run=False):
    if run:
        ped_df = pd.read_csv(ped_csv)
        frame_list = pd.unique(ped_df['frame'])
        max_frame = int(frame_list[-1])
        last_frame = ped_df[ped_df['frame'] == max_frame]
        last_frame = last_frame[last_frame['joint_type'] == 98]
        location_x = float(last_frame[last_frame['joint_type'] == 98]['3D_x'])
        location_y = float(last_frame[last_frame['joint_type'] == 98]['3D_y'])
        location_z = float(float(last_frame[last_frame['joint_type'] == 98]['3D_z']) + 1.5)
        print(location_x)
        print(location_y)
        print(location_z)
        try:
            with open(freefall_location, 'a') as file:
                file.write(str(location_x) + ',')
                file.write(str(location_y) + ',')
                file.write(str(location_z) + ',')
                file.write('\n')
        except FileNotFoundError:
            pass


if __name__ == '__main__':

    GTA_path = r'D:\GTAV'
    seq_folder = os.path.join(r'D:\GTAV', 'zips')
    img_folder = os.path.join(seq_folder, 'seq_00000003')
    ped_csv_path = os.path.join(img_folder, 'peds.csv')
    log_path = r'D:\GTAV'

    store_file = open(os.path.join(log_path, '2results.txt'), 'a')
    store_file.truncate(0)
    '''for folder in os.listdir(seq_folder):
        seq_error = 1
        result = 0
        try:
            index = int(folder.rsplit('_', 1)[1])

            if 0 <= index <= 100000:

                img_folder = os.path.join(seq_folder, folder)
                ped_csv_path = os.path.join(img_folder, 'peds.csv')
                try:
                    result = int(analysis_peds_csv(ped_csv_path))

                    string = folder + '  ' + str(result)
                    store_file.write(string)
                    store_file.write('\n')
                    print(string)
                    seq_error = 0
                except FileNotFoundError:
                    string = folder + '...........Not found'
                    store_file.write(string)
                    store_file.write('\n')
                    print(string)
                except TypeError:
                    string = folder + '...........TypeError'
                    store_file.write(string)
                    store_file.write('\n')
                    print(string)
                except ValueError:
                    continue

        except IndexError:
            pass
        except ValueError:
            pass
    '''
    list = []
    i = 0
    for files in os.listdir(r'D:/GTAV/csvs'):
        i += 1
        try:
            result = analysis_peds_csv(os.path.join(r'D:/GTAV/csvs', files))
            if result == 1:
                store_file.write(files[:-9] + '\n')
                print(files[:-9])
        except:
            continue

    store_file.close()

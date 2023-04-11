'''camera_moving? save_depth? save_stencil? random_ped?
weather
daytime
location_type
center_coord_world .x.y.z(that camera focus on)
camera_rtation_world .x.y.z
camera_displacement .x.y.z

ped_number

ped_1_action(name dict time)
ped_1_init_displacement .x.y.z heading hash type (if random_ped==False)

ped_2_action(name dict time)
ped_2_init_displacement .x.y.z heading hash type (if random_ped==False)


...
'''

import random
import os
import numpy as np
import ntpath
from angle_generator import generate_1_angle
from view_3d_map import get_location_set, get_a_location
import json

setting_list = [0, 1, 1, 1]  # first line, 1 True 0 False

information_list_sample = ['vid_00011111.txt', 'CLEAR', [17, 59, 1],
                           'Suburb', [1020.518, 663.270, 153.516],
                           [-10.09855833792897, 0.0, -11.788561927397344],
                           [-1.4204661872745872, -6.806175577483673, 1.2383059444666957],
                           1,  # ped number
                           [['ac_ig_3_p3_b-1 player_two_dual-1 6000', [-0.5, -0.5, 0.5], -180], []]  # action
                           ]

# each sequence vid_{08d}.txt
# weather type (day & night)
# time hour, min, sec   day time 7 <= h <= 19
# location type
# location
# camera rotation
# camera displacement

from gen_config3 import get_position_with_label
from gen_config3 import retrieve_labeled_position


def gen_scenario_info(seq_list, ped_num, config, action_file, l_pt_set):
    weathers_day = ["CLEAR", "EXTRASUNNY", "RAIN", "THUNDER", "CLOUDS", "OVERCAST", "SMOG", "FOGGY", "XMAS", "BLIZZARD",
                    "CLEAR", "CLEAR", "CLEAR", "EXTRASUNNY", "EXTRASUNNY"]  # to balance weightage
    weathers_night = ["CLEAR", "EXTRASUNNY", "RAIN", "THUNDER", "SMOG", "FOGGY", "XMAS", "BLIZZARD",
                      "CLEAR", "CLEAR", "CLEAR", "CLEAR", "EXTRASUNNY", "EXTRASUNNY", "EXTRASUNNY"]
    for seq in seq_list:
        info_list = [seq]
        time_h = random.randint(0, 23)
        time_m = random.randint(0, 59)
        time_s = random.randint(0, 59)

        if 7 <= time_h <= 19:
            weather = random.choice(weathers_day)
        else:
            weather = random.choice(weathers_night)

        info_list.append(weather)
        info_list.append([time_h, time_m, time_s])

        x, y, l_type = get_a_location(l_pt_set, l_type=None)

        location = [x, y, 0]
        '''if ped_num != 1:
                   file_list = os.listdir(r'D:\GTAV\pending_upload')

            file_name = random.choice(file_list)
            secne_f = file_name
            location = os.path.join(r'D:\GTAV\pending_upload', file_name)'''


        info_list.append(l_type)
        info_list.append(location)

        camera_angle, camera_dis = get_camera()
        info_list.append(camera_angle)
        info_list.append(camera_dis)

        info_list.append(ped_num)

        if ped_num == 1:
            seq_num = seq[8:12]
            action = get_line(action_file, seq_num)
            ped_info = [action[:-2], [random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), random.uniform(1, 1.5)],
                        -180]
            info_list.append(ped_info)
            write_scenario(config, info_list, scenario_dir=r'D:\GTAV\MTA-scenarios-nu')

        min_len_set = random.choice([2600, 4000, 5600, 7000])
        if ped_num != 1:
            seq_num = seq[8:12]
            for i in range(ped_num):
                action = get_rand_line(action_file, min_len=min_len_set)
                ped_info = [action[:-2], [random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), random.uniform(1, 1.5)],
                            random.uniform(-180, 180)]
                info_list.append(ped_info)
            write_scenario(config, info_list, scenario_dir=r'D:\GTAV\MTA-scenarios-nu')




def get_camera():
    rz, rx = generate_1_angle()
    distance = random.uniform(5.0, 10.0)
    distance = 8
    z_off = distance * np.sin(rx)
    x_off = distance * np.cos(rx) * np.sin(rz)
    y_off = -distance * np.cos(rx) * np.cos(rz)
    ry = 0.0
    rx = rx * 180 / np.pi
    rz = rz * 180 / np.pi

    return [rx, ry, rz], [x_off, y_off, -z_off]


def get_line(filepath, line_count):
    file = open(filepath, 'r').readlines()
    return file[int(line_count) - 1]


def get_rand_line(filepath, min_len=1000):
    file = open(filepath, 'r').readlines()
    f_len = len(file)
    action_len = 0
    line_count = random.randint(0, f_len)
    while action_len <= min_len:
        line_count = random.randint(0, f_len -1 )
        action_len = float(file[int(line_count)].rsplit(' ', 2)[2])
    return file[int(line_count)]


def write_scenario(setting, info, scenario_dir):
    os.makedirs(scenario_dir, exist_ok=True)
    setting = ' '.join(str(ele) for ele in setting)

    with open(os.path.join(scenario_dir, info[0]), 'a') as f:
        f.truncate(0)
        f.write(setting)
        f.write('\n')
        f.write(info[1])
        f.write('\n')

        daytime = ' '.join(str(ele) for ele in info[2])
        f.write(daytime)
        f.write('\n')
        f.write(info[3])
        f.write('\n')

        location = ' '.join(str(ele) for ele in info[4])
        f.write(location)
        f.write('\n')

        camera_rotation = ' '.join(str(ele) for ele in info[5])
        f.write(camera_rotation)
        f.write('\n')
        camera_displacement = ' '.join(str(ele) for ele in info[6])
        f.write(camera_displacement)
        f.write('\n')

        ped_num = int(info[7])
        f.write(str(ped_num))
        f.write('\n')

        for i in range(ped_num):
            action = info[8 + i][0]
            f.write(action + '0')
            f.write('\n')

            ped_displacement = ' '.join(str(ele) for ele in info[8 + i][1])
            f.write(ped_displacement)

            heading = info[8 + i][2]
            f.write(' ' + str(heading))

            try:
                p_hash = info[8 + i][3]
                p_type = info[8 + i][4]
                f.write(' ' + p_hash)
                f.write(' ' + p_type)
            except IndexError:
                pass
            f.write('\n')

    print('Scenario ' + info[0] + ' created.')


if __name__ == '__main__':
    animation_path = os.path.join(r'D:\GTAV', 'PedAnimList_valid_timed_filter.txt')

    list_test = []
    batch_num = 0
    with open(r'C:\Users\it_admin\Desktop\mta-2021-1210\python\data_manage\status.json', 'r') as f:
        status = json.load(f)

    for i in range(100000, 120000):
        j = i + 1 + batch_num*100000
        seq = 'seq_{:08d}'.format(j)
        try:
            if status[seq] in ['rejected mesh not loaded', 'rejected camera underground']:
                pass
            else:
                pass

        except KeyError:
            pass

        list_test.append('vid_{:08d}.txt'.format(j))

    pt_set = get_location_set(r'GTAV-MAP-segmented.jpg')
    # gen_scenario_info(list_test, ped_num=1, action_file=animation_path, config=[0, 1, 1, 1], l_pt_set=pt_set)
    # write_scenario(setting_list, information_list_sample, scenario_dir=r'D:\GTAV\MTA-scenarios')

    # list_test = ['seq_00000001.txt']
    filtered_animation_path = r'D:\GTAV\valid_action.txt'
    gen_scenario_info(list_test, ped_num=random.randint(3, 6), action_file=filtered_animation_path, config=[0, 1, 1, 1], l_pt_set=pt_set)

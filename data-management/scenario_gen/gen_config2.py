import random
import os
import numpy as np
import re
import pickle


places_per_action = 1
camera_per_place = 1
start = 50  # which motion to start 6600x2
ani_num = 10000
current_path = os.path.dirname(os.path.abspath(__file__))  # 目前工作目录
# locationlist_path = os.path.join(current_path,'locationList.txt')  #读取 locationlist 目录
animation_path = os.path.join(current_path, 'PedAnimList.txt')
game_path = r'D:\GTAV'

Local_Type = ['HighLocations', 'offices', 'interiors', 'Misc', 'shopping locations', 'Indoor', 'Outdoors', 'Landmarks',
              'apartment interiors', 'Others']
local_type_list = ['shopping locations', 'Outdoors', 'Landmarks', 'Others']

# if (not os.path.exists(locationlist_path)):
#     print("Can't find locationList.txt!")
#     exit(0)
# else:
#     with open(locationlist_path, 'r') as f:
#         local_coords = []
#         for line in f.readlines():
#             local_name, local_type, x, y, z = line.split(',')
#             x = float(x)
#             y = float(y)
#             z = float(z)
#             if -5000<x<5000 and -5000<y<8000 and 0<z<500 and local_type in local_type_list:
#                 local_coords.append([x, y, z])

if (not os.path.exists(game_path)):
    print("Can't find Game path!")
    exit(0)

CAM_FOV = 50
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

# default options of places in MOD
places = {"MICHAEL'S HOUSE": (-852.4, 160.0, 65.6),
          "FRANKLIN'S HOUSE": (7.9, 548.1, 175.5),
          "TREVOR'S TRAILER": (1985.7, 3812.2, 32.2),
          # "AIRPORT ENTRANCE" :(-1034.6, -2733.6, 13.8),
          # "AIRPORT FIELD": (-1336.0, -3044.0, 13.9) ,
          "ELYSIAN ISLAND": (338.2, -2715.9, 38.5),
          "JETSAM": (760.4, -2943.2, 5.8),
          "STRIPCLUB": (127.4, -1307.7, 29.2),
          "ELBURRO HEIGHTS": (1384.0, -2057.1, 52.0),
          "FERRIS WHEEL": (-1670.7, -1125.0, 13.0),
          "CHUMASH": (-3192.6, 1100.0, 20.2),
          "WINDFARM": (2354.0, 1830.3, 101.1),
          "MILITARY BASE": (-2047.4, 3132.1, 32.8),
          "MCKENZIE AIRFIELD": (2121.7, 4796.3, 41.1),
          "DESERT AIRFIELD": (1747.0, 3273.7, 41.1),
          "CHILLIAD": (425.4, 5614.3, 766.5)}


# places = list(places.values())
# places = local_coords

def cosine_dist_2d(x1, y1, z1, x2, y2, z2):
    return (x1 * x2 + y1 * y2 + z1 * z2) / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) *
                                            np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))


def gen_cam(x, y, z, xyz_off=None, xyz_dis=None, angle_limit=75):
    '''
    angle_limit 最大俯视视角
    '''
    max_z_off = 0.3
    if xyz_off is not None:
        x_off, y_off, z_off = xyz_off
    else:
        if xyz_dis is None or xyz_dis <= 0:
            x_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            y_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            z_off = random.uniform(2.0, 6.0)
        elif xyz_dis > 0:
            z_off = random.uniform(-xyz_dis * np.sin(abs(angle_limit) * np.pi / 180.), max_z_off)
            x_off = random.uniform(0, xyz_dis * 0.5) * random.choice([-1, 1])
            while xyz_dis ** 2 - x_off ** 2 - z_off ** 2 < 0:
                z_off = random.uniform(-xyz_dis * np.sin(abs(angle_limit) * np.pi / 180.), max_z_off)
                x_off = random.uniform(0, xyz_dis * 0.5) * random.choice([-1, 1])
            y_off = (xyz_dis ** 2 - x_off ** 2 - z_off ** 2) ** 0.5 * random.choice([-1, 1])

    ry = 0.0
    rx = np.arcsin(z_off / np.sqrt(x_off ** 2 + y_off ** 2 + z_off ** 2)) * 180 / np.pi
    rz = np.arcsin(-x_off / np.sqrt(x_off ** 2 + y_off ** 2 + z_off ** 2)) * 180 / np.pi
    if y_off < 0:
        if rz < 0:
            rz = -rz - 180
        elif rz > 0:
            rz = -rz + 180
    return -x_off, -y_off, -z_off, rx, ry, rz


def gen_var(xyz, xyz_off=None, xyz_dis=7, angle_limit=75, ped_num=1, behaviour=1, a_num=0):
    px, py, pz = xyz
    moving = 0
    stop = 0
    group = 0
    speed = 1.0
    go_from = (px, py, pz)
    go_to = (px, py, pz)
    if behaviour == 0:
        btype = random.randint(0, 13)
    else:
        btype = 0
    radius = 20

    cx, cy, cz, rx, ry, rz = gen_cam(px, py, pz, xyz_off=None, xyz_dis=xyz_dis, angle_limit=angle_limit)

    line1 = '{} {} {} {} {} {} {} {} {}\n'.format(moving, cx, cy, cz, stop, rx, ry, rz, a_num)
    line2 = '{} {} {} {} {} {}\n'.format(px, py, pz, px, py, pz)
    line3 = '{} {} {} {} {} {}\n'.format(px, py, pz, px, py, pz)
    if ped_num > 0:
        line4 = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
            ped_num, px, py, pz, group, behaviour, speed, px, py, pz, px, py, pz, 1000000, btype, radius, 1, 1, 1)
    else:
        line4 = ''
        line3 = line3[:-1]

    return line1 + line2 + line3 + line4


'''
Main processing part
'''
if __name__ == '__main__':
    anim_list = open(animation_path, 'r').readlines()
    anim_set = open(os.path.join(current_path, 'PedAnimSet.txt'), 'r').readlines()
    #  anim_set = open(os.path.join(r'D:\GTAV', 'invalid_list.txt'), 'r').readlines()
    anim_set = [an[:-1] for an in anim_set]

    with open(game_path + '//PedAnimList.txt', 'w') as f:
        for a in anim_list:
            ani_type = a.split(' ')[0]
            if ani_type in anim_set[start:start + ani_num]:
                f.write(a)

    places_num = ani_num * places_per_action
    places = []
    for _ in range(places_num):
        i = random.randint(-3000, 3000)
        j = random.randint(-3000, 3000)
        places.append(['grid_{}_{}'.format(i, j), '', i, j, random.randint(30, 80)])
    # places = places
    _places = []
    random.shuffle(places)
    for k in places:
        for i in range(camera_per_place):
            _places.append(k)
    places = _places

    print(f'#actions: {ani_num}, #places: {places_num}')

    count = start-1
    with open(r'D:\GTAV\invalid_seq_list.txt', 'rb') as fp:
        invalid_list = pickle.load(fp)

    for anum in range(ani_num):

        for p in range(places_per_action):
            px, py, pz = places[anum * places_per_action + p][-3:]
            for i in range(camera_per_place):
                textname = 'vid_%08d.txt' % count
                count += 1
                behaviour_choice = -1
                ped_num = 1
                # if i == 0: 
                #     ped_num = 1
                # else:
                #     ped_num = 0
                if anum not in invalid_list:
                    pass

                with open(os.path.join(game_path + '//MTA-Scenarios', textname), 'w') as f:
                    f.write(gen_var(xyz=(px, py, pz), xyz_off=None, xyz_dis=random.randint(5, 7), angle_limit=45,
                                    ped_num=ped_num, behaviour=behaviour_choice, a_num=anum))

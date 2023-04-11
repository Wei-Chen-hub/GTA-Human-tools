import argparse
import random
import os
import os.path as osp
import math
import numpy as np
from tqdm import trange

CAM_FOV = 50
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

places = [["MICHAEL'S HOUSE", "", -852.4, 160.0, 65.6],
          ["FRANKLIN'S HOUSE", "", 7.9, 548.1, 175.5],
          ["TREVOR'S TRAILER", "", 1985.7, 3812.2, 32.2],
          ["AIRPORT ENTRANCE", "", -1034.6, -2733.6, 13.8],
          ["AIRPORT FIELD", "", -1336.0, -3044.0, 13.9],
          ["ELYSIAN ISLAND", "", 338.2, -2715.9, 38.5],
          ["JETSAM", "", 760.4, -2943.2, 5.8],
          ["STRIPCLUB", "", 127.4, -1307.7, 29.2],
          ["ELBURRO HEIGHTS", "", 1384.0, -2057.1, 52.0],
          ["FERRIS WHEEL", "", -1670.7, -1125.0, 13.0],
          ["CHUMASH", "", -3192.6, 1100.0, 20.2],
          ["WINDFARM", "", 2354.0, 1830.3, 101.1],
          ["MILITARY BASE", "", -2047.4, 3132.1, 32.8],
          ["MCKENZIE AIRFIELD", "", 2121.7, 4796.3, 41.1],
          ["DESERT AIRFIELD", "", 1747.0, 3273.7, 41.1],
          ["CHILLIAD", "", 425.4, 5614.3, 766.5]]


def get_normal_vector_2d(x, y):
    if (y == 0 and x == 0):
        return 0, 0
    elif (y == 0):
        return 0, x
    else:
        return 1, -x / y


def cosine_dist_2d(x1, y1, x2, y2):
    return (x1 * x2 + y1 * y2) / (math.sqrt(x1**2 + y1**2) *
                                  math.sqrt(x2**2 + y2**2))


def cal_pitch(x_off, y_off, z_off):
    return -(math.acos(
        math.sqrt(x_off**2 + y_off**2) /
        math.sqrt(x_off**2 + y_off**2 + z_off**2)) / math.pi) * 180


def gen_cam(x, y, z, xyz_off=None, xyz_dis=None, random_shift=True):
    if xyz_off is not None:
        x_off, y_off, z_off = xyz_off
    else:
        if xyz_dis is None:
            x_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            y_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            z_off = random.uniform(2.0, 6.0)
        else:
            z_off = random.uniform(1.0, xyz_dis * 0.5)
            x_off = random.uniform(0, xyz_dis * 0.5) * random.choice([-1, 1])
            y_off = (xyz_dis**2 - x_off**2 - z_off**2)**0.5
            x_off = 0.00000001
            z_off = random.uniform(1.0, xyz_dis * 0.9)
            y_off = (xyz_dis**2 - z_off**2)**0.5

    h_angle = (math.acos(cosine_dist_2d(-x_off, -y_off, 0, 1)) /
               math.pi) * 180 * (x_off / abs(x_off))
    rx = cal_pitch(x_off, y_off, z_off)
    ry = 0.0
    rz = h_angle
    if random_shift:
        rz += random.uniform(-25, 25)
    return 0, x_off, y_off, z_off, 0, rx, ry, rz


def test_gen_cam():
    px, py, pz = (1985.541, 3812.397, 32.218)
    cx, cy, cz = (1982.266, 3810.265, 35.003)
    rot = (-29.249, 0, -62.803)
    px, py, pz = (1749.999, 2377.251, 41.107)
    cx, cy, cz = (1746.134, 3279.546, 41.858)
    rot = (-1.831, 0, -125.808)
    px, py, pz = (-3192.6, 1100, 20.192)
    cx, cy, cz = (-3196.561, 1102.095, 20.391)
    rot = (5.242, 0, -122.997)
    res = gen_cam(px, py, pz, (cx - px, cy - py, cz - pz))[-3:]
    print(rot)
    print(res)


def projection(x, y, z, cam_x, cam_y, cam_z, rot_x, rot_y, rot_z):
    x -= cam_x
    y -= cam_y
    z -= cam_z
    rx = rot_x / 180. * math.pi
    ry = rot_y / 180. * math.pi
    rz = rot_z / 180. * math.pi
    cx = math.cos(rx)
    cy = math.cos(ry)
    cz = math.cos(rz)
    sx = math.sin(rx)
    sy = math.sin(ry)
    sz = math.sin(rz)
    dx = cy * (sz * y + cz * x) - sy * z
    dy = sx * (cy * z + sy * (sz * y + cz * x)) + cx * (cz * y - sz * x)
    dz = cx * (cy * z + sy * (sz * y + cz * x)) - sx * (cz * y - sz * x)
    fov_rad = CAM_FOV * math.pi / 180.0
    f = (SCREEN_HEIGHT / 2.0) * math.cos(fov_rad / 2.0) / math.sin(
        fov_rad / 2.0)
    res_x = (dx * (f / dy)) / SCREEN_WIDTH + 0.5
    res_y = (0.5 - (dz * (f / dy)) / SCREEN_HEIGHT)
    res_x *= SCREEN_WIDTH
    res_y *= SCREEN_HEIGHT
    return res_x, res_y


def fov_check(x, y, z, cx, cy, cz, rx, ry, rz):
    tx, ty = projection(x, y, z, cx, cy, cz, rx, ry, rz)
    if tx > SCREEN_WIDTH * 0.1 and tx < SCREEN_WIDTH * 0.9 and ty > SCREEN_HEIGHT * 0.1 and ty < SCREEN_HEIGHT * 0.9:
        return True
    else:
        return False


def gen_peds_str(
    x,
    y,
    z,
    cx,
    cy,
    cz,
):
    is_combat = (random.random() > 0.5)
    gen_call_num = random.randint(2, 8) // 2
    if is_combat and random.random() > 0.5:
        gen_call_num -= 1
    pos_str = ''
    for _ in range(gen_call_num):
        pos_str += gen_positive_str(x, y, z, cx, cy, cz)
    if (not is_combat):
        return pos_str
    behavior = 6
    n = random.randint(2, 5)
    speed = 1.0
    task_time = 1000000
    _type = 0 if behavior != 0 else random.randint(0, 14)
    radious = random.randint(10, 30)
    minimal_length = 1
    time_between_walks = 1
    spawning_radius = 1
    combat_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n, x, y, z, 0, behavior, speed, x, y, z, x, y, z, task_time, _type,
        radious, minimal_length, time_between_walks, spawning_radius)
    return combat_str + pos_str


def gen_single_ped_str(x, y, z, cx, cy, cz):
    behavior = -1
    n = 1
    speed = 1.0
    task_time = 1000000
    _type = 0
    radious = random.randint(10, 30)
    minimal_length = 1
    time_between_walks = 1
    spawning_radius = 1
    ped_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n, x, y, z, 0, behavior, speed, x, y, z, x, y, z, task_time, _type,
        radious, minimal_length, time_between_walks, spawning_radius)
    return ped_str


def gen_climb_peds_str(x1, y1, z1, x2, y2, z2, cx, cy, cz):
    pos_str = ''
    behavior = 9
    n = random.randint(1, 2)
    speed = 1.0
    task_time = 1000000
    radious = random.randint(10, 30)
    _type = 0
    minimal_length = 1
    time_between_walks = 1
    spawning_radius = 1
    combat_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n, x1, y1, z1, 0, behavior, speed, x1, y1, z1, x2, y2, z2, task_time,
        _type, radious, minimal_length, time_between_walks, spawning_radius)
    return combat_str + pos_str


def get_coord(idx, places):
    n_places = len(places)
    _id = idx if idx is not None else random.randint(0, n_places - 1)
    return places[_id][2], places[_id][3], places[_id][4]


def get_coord_p2p(idx, places):
    n_places = len(places)
    _id = idx if idx is not None else random.randint(0, n_places - 1)
    return places[_id]


def gen_positive_str(x, y, z, cx, cy, cz):
    x += random.uniform(-1, 1) * 5
    y += random.uniform(-1, 1) * 5
    n_move = random.randint(0, 5)
    n_stand = random.randint(0, 6) // 2
    behavior = 6
    while (behavior == 6 or behavior == 8):
        behavior = random.randint(0, 7)
    speed = 1.0
    task_time = 1000000
    _type = 0 if behavior != 0 else random.randint(0, 14)
    radious = random.randint(10, 30)
    minimal_length = 1
    time_between_walks = 1
    spawning_radius = 1
    stand_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n_stand, x, y, z, 0, behavior, speed, x, y, z, x, y, z, task_time,
        _type, radious, minimal_length, time_between_walks, spawning_radius)
    x += random.uniform(-1, 1) * 3
    y += random.uniform(-1, 1) * 3
    x_n, y_n = get_normal_vector_2d(x - cx, y - cy)
    l = math.sqrt(x_n**2 + y_n**2)
    if (l != 0):
        x_n /= l
        y_n /= l
    move_radius = random.uniform(10, 15)
    move_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n_move, x, y, z, 0, 8, speed, -move_radius * x_n, -move_radius * y_n,
        0, move_radius * x_n, move_radius * y_n, 0, task_time, _type, radious,
        minimal_length, time_between_walks, spawning_radius)
    return stand_str + move_str


def gen_config(filename,
               places,
               loc_idx=None,
               xyz_off=None,
               peds_str_fn=gen_peds_str,
               **kwargs):
    x, y, z = get_coord(loc_idx, places)
    with open(filename, "w") as f:
        moving, cx, cy, cz, stop, rx, ry, rz = gen_cam(x,
                                                       y,
                                                       z,
                                                       xyz_off=xyz_off)
        f.write("{} {} {} {} {} {} {} {}\n".format(moving, cx, cy, cz, stop,
                                                   rx, ry, rz))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write(peds_str_fn(x, y, z, cx, cy, cz))


def gen_single_config(filename,
                      places,
                      loc_idx=None,
                      xyz_off=None,
                      places_per_action=1,
                      camera_per_place=1,
                      peds_str_fn=gen_single_ped_str,
                      **kwargs):
    x, y, z = get_coord(loc_idx, places)
    with open(filename, "w") as f:
        moving, cx, cy, cz, stop, rx, ry, rz = gen_cam(x,
                                                       y,
                                                       z,
                                                       xyz_off=xyz_off,
                                                       xyz_dis=8,
                                                       random_shift=False)
        f.write("{} {} {} {} {} {} {} {} {}\n".format(
            moving, cx, cy, cz, stop, rx, ry, rz,
            (loc_idx // (places_per_action * camera_per_place))))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write(peds_str_fn(x, y, z, cx, cy, cz))


def gen_p2p_config(filename, places, loc_idx=None, xyz_off=None, **kwargs):
    x1, y1, z1, x2, y2, z2 = get_coord_p2p(loc_idx, places)
    x, y, z = x1, y1, z1
    with open(filename, "w") as f:
        moving, cx, cy, cz, stop, rx, ry, rz = gen_cam(x1,
                                                       y1,
                                                       z1,
                                                       xyz_off=xyz_off)
        f.write("{} {} {} {} {} {} {} {}\n".format(moving, cx, cy, cz, stop,
                                                   rx, ry, rz))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write("{} {} {} {} {} {}\n".format(x, y, z, x, y, z))
        f.write(gen_climb_peds_str(x1, y1, z1, x2, y2, z2, cx, cy, cz))


def get_loc_list(filename):
    locs = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            paras = line.strip().split(',')
            locs.append([
                paras[0], paras[1],
                float(paras[2]),
                float(paras[3]),
                float(paras[4])
            ])
    return locs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Config generation for GTA data collection')
    parser.add_argument(
        '--places_per_action',
        default=1,
        type=int,
        help='Each action will be performed in N different places. Default: 1.'
    )
    parser.add_argument(
        '--camera_per_place',
        default=1,
        type=int,
        help='Each place will contain N different cameras. Default: 1.')
    parser.add_argument('--dst_dir', default='generated_senarios', type=str)
    parser.add_argument('--anim_list',
                        default='PedAnimList_valid_timed_filter_example.txt',
                        type=str)
    parser.add_argument('--use_locationlist', action='store_true')
    args = parser.parse_args()

    anim_list = open(args.anim_list, 'r').readlines()
    anim_list = set([k.split(' ')[0] for k in anim_list])
    anim_num = len(anim_list)

    current_path = osp.dirname(osp.abspath(__file__))
    locationlist_path = osp.join(current_path, 'locationList.txt')

    if False:
    # if args.use_locationlist:
        assert osp.exists(locationlist_path), f'{locationlist_path} is not found'
        places = get_loc_list(locationlist_path)
        places_num = len(places)
    else:
        places_num = anim_num * args.places_per_action
        places = []
        for _ in range(places_num):
            i = random.randint(-1000, 1000)
            j = random.randint(-2000, 500)
            places.append(['grid_{}_{}'.format(i, j), '', i, j, 150])
        # places = places
        _places = []
        random.shuffle(places)
        for k in places:
            for i in range(args.camera_per_place):
                _places.append(k)
        places = _places

    print(f'#actions: {anim_num}, #places: {places_num}')

    #test_gen_cam()

    os.makedirs(args.dst_dir, exist_ok=True)

    for i in trange(places_num):
        gen_single_config(r"{}/vid_{:08d}.txt".format(args.dst_dir, i),
                          places,
                          i,
                          places_per_action=args.places_per_action,
                          camera_per_place=args.camera_per_place)

import random
import os
import numpy as np
import ntpath
from angle_generator import generate_1_angle

game_path = r'D:\GTAV'
peds_num = 5
places_per_action = 1
camera_per_place = 1
start = 1  # which motion to start 6600x2
ani_num = 56639

current_path = os.path.dirname(os.path.abspath(__file__))  # 目前工作目录
locationlist_path = os.path.join(current_path, 'locationList.txt')  # 读取 locationlist 目录
new_locationlist_path = r'D:\GTAV\1location 1-5000.txt'
# animation_path = os.path.join(current_path, 'PedAnimList.txt')
animation_path = os.path.join(r'D:\GTAV', 'PedAnimList_valid_timed_filter.txt')
new_animation_path = r'D:\GTAV\1action_list 1-5000.txt'
invalid_animation_path = r'D:\GTAV\invalid_list.txt'
output_path = os.path.join(game_path, 'MTA-Scenarios')

location_type_list = ['city_non_street', 'coast', 'roof', 'wild', 'special', 'indoor']
labeled_location_path = os.path.join(r'C:\Users\12595\Desktop\mta-98', 'locations.txt')
city_street_locations = r'C:\Users\12595\Desktop\mta-98-0505\mta-98\locations_city_street.txt'

try:
    os.makedirs(output_path)

except FileExistsError as e:
    pass


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def get_random_line(filepath):
    file = open(filepath, 'r')
    line = next(file)
    for num, aline in enumerate(file, 2):
        if random.randrange(num):
            continue
        line = aline
    return line


def get_line(filepath, line_count):
    file = open(filepath, 'r').readlines()
    return file[int(line_count) - 1]


def get_random_position(location_file):
    line = get_random_line(location_file)
    split = line.split(',')
    location = []
    if split:
        for item in split:
            try:
                a = float(item)
                location.append(item)
            except ValueError:
                pass

        return location[0], location[1], location[2].rsplit('\n', 1)[0]

    else:
        print('Location data not found')


def retrieve_labeled_position(location_file):
    with open(location_file, 'r') as f:
        lines = f.readlines()
        type_count = 0

        # different types of terrain
        city_non_street = []
        wild = []
        coast = []
        roof = []
        special = []
        indoor = []

        for line in lines:

            if 'list' in line:
                type_count += 1
                continue

            try:
                locx = line.rsplit(' ', 3)[0]
                locy = line.rsplit(' ', 3)[1]
                locz = line.rsplit(' ', 3)[2]
            except:
                try:
                    locx = line.rsplit(',', 3)[0]
                    locy = line.rsplit(',', 3)[1]
                    locz = line.rsplit(',', 3)[2]
                except:
                    continue
            if type_count == 1:
                city_non_street.append([locx, locy, locz])
            if type_count == 2:
                wild.append([locx, locy, locz])
            if type_count == 3:
                coast.append([locx, locy, locz])
            if type_count == 4:
                roof.append([locx, locy, locz])
            if type_count == 5:
                special.append([locx, locy, locz])
            if type_count == 6:
                indoor.append([locx, locy, locz])

    print('successfully retrieved labeled location')

    location_dict = {}
    for types in [city_non_street, wild, coast, roof, special, indoor]:
        name = namestr(types, locals())[0]
        location_dict[name] = types
    print(location_dict)
    return location_dict


def get_position_with_label(location_dict, wanted_label):
    location_random = random.choice(location_dict[wanted_label])
    locx = location_random[0]
    locy = location_random[1]
    locz = location_random[2]
    return locx, locy, locz


def get_certain_position(location_file, line):
    line = get_line(location_file, line)
    split = line.split(',')
    location = []
    if split:
        for item in split:
            try:
                a = float(item)
                location.append(item)
            except ValueError:
                pass

        return location[0], location[1], location[2]

    else:
        print('Location data not found')


def get_random_action(peds_num, action_file):
    i = 0
    actions = []
    while i < peds_num:
        temp = get_random_line(action_file)
        temp = temp.replace('\n', '')
        print(temp)
        if not temp:
            pass
        else:
            temp = temp.rsplit(' ', 2)
            anim_dict = temp[0]
            anim_name = temp[1]
            anim_time = temp[2]
            string = anim_dict + ' ' + anim_name + ' ' + anim_time
            actions.append(string)
            i += 1
    if not actions:
        print('Action name not found')
    else:
        return actions


def gen_cam(x, y, z, xyz_off=None, xyz_dis=None, angle_limit=75, mode='angle_generator'):
    '''
    angle_limit 最大俯视视角
    mode: horizontal / look_down / angle_generator
    '''
    max_z_off = 0.3

    if xyz_off is not None:
        x_off, y_off, z_off = xyz_off
    else:
        if xyz_dis is None or xyz_dis <= 0:
            x_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            y_off = random.uniform(0.5, 12.0) * random.choice([-1, 1])
            if mode == 'horizontal':
                z_off = random.uniform(-1.0, 1.0)
            elif mode == 'high':
                z_off = random.uniform(1.0, 6.0)
        elif xyz_dis > 0:
            z_off = random.uniform(-xyz_dis * np.sin(abs(angle_limit) * np.pi / 180.), max_z_off)
            x_off = random.uniform(0, xyz_dis * 0.5) * random.choice([-1, 1])
            while xyz_dis ** 2 - x_off ** 2 - z_off ** 2 < 0:
                z_off = random.uniform(-xyz_dis * np.sin(abs(angle_limit) * np.pi / 180.), max_z_off)
                x_off = random.uniform(0, xyz_dis * 0.5) * random.choice([-1, 1])
            y_off = (xyz_dis ** 2 - x_off ** 2 - z_off ** 2) ** 0.5 * random.choice([-1, 1])
    try:
        ry = 0.0
        rx = np.arcsin(z_off / np.sqrt(x_off ** 2 + y_off ** 2 + z_off ** 2)) * 180 / np.pi
        rz = np.arcsin(-x_off / np.sqrt(x_off ** 2 + y_off ** 2 + z_off ** 2)) * 180 / np.pi
        if y_off < 0:
            if rz < 0:
                rz = -rz - 180
            elif rz > 0:
                rz = -rz + 180
        return -x_off, -y_off, -z_off, rx, ry, rz
    except UnboundLocalError:
        pass

    if mode == 'angle_generator':
        rx = -10
        # while rx <= -0.3:
        rz, rx = generate_1_angle()
        # rz = 0
        # rx = 5.369 * np.pi / 180
        # rz = 28.567 * np.pi / 180

        distance = random.uniform(5.0, 8.0)
        print('distance = ', distance)
        z_off = distance * np.sin(rx)
        x_off = distance * np.cos(rx) * np.sin(rz)
        y_off = -distance * np.cos(rx) * np.cos(rz)
        print('x_off = ', x_off)
        print('z_off = ', z_off)
        ry = 0.0
        rx = rx * 180 / np.pi
        rz = rz * 180 / np.pi

        return x_off, y_off, -z_off, rx, ry, rz


def create_scenario(filepath, location_file, peds_num, action_file, mode):
    file = open(filepath, 'a')
    file.truncate(0)
    file.write(str(peds_num))
    file.write('\n')
    locz = 100
    if not mode:
        while float(locz) >= 50:
            locx, locy, locz = get_random_position(location_file)
            locz = locz[:-1]
            location_type = 'city_street'
        if float(locz) >= 50:
            file.truncate(0)
            file.close()
            return 0
    try:
        # if mode.is_integer():
        locx, locy, locz = get_certain_position(location_file, mode)
        locz = locz[:-1]
        location_type = 'city_street'
    except AttributeError:
        pass
    except ValueError:
        pass

    if mode == 'random_labeled':
        mode = random.choice(location_type_list)
        temp_list = [mode, mode, 'city_street']
        location_type = random.choice(temp_list)
        if location_type == 'city_street':
            locx, locy, locz = get_random_position(city_street_locations)
        else:
            locx, locy, locz = get_position_with_label(location_dict, mode)
    elif mode in location_type_list:
        location_type = mode
        locx, locy, locz = get_position_with_label(location_dict, mode)

    # locx, locy, locz = str(-3022.2), str(39.968), str(13.611)
    # location_type = "city_non_street"

    file.write(locx + ' ')
    file.write(locy + ' ')
    file.write(locz + ' ')
    file.write(location_type)
    file.write('\n')

    cx, cy, cz, rx, ry, rz = gen_cam(x=locx, y=locy, z=locz)
    # adjust camera distance according to ped number
    # adjust_const = 1
    if location_type in ['city_street', 'city_non_street', 'coast', 'wild']:
        adjust_const = float(peds_num) * 0.1 + 0.6
    elif location_type in ['roof', 'special', 'indoor']:
        adjust_const = float(peds_num) * 0.05 + 0.4
    cx, cy, cz = cx * adjust_const, cy * adjust_const, cz * adjust_const

    file.write(str(cx) + ' ')
    file.write(str(cy) + ' ')
    file.write(str(cz))
    file.write('\n')
    file.write(str(rx) + ' ')
    file.write(str(ry) + ' ')
    file.write(str(rz))
    file.write('\n')

    if peds_num == 1:
        head, tail = ntpath.split(filepath)
        temp = tail.rsplit('_', 1)[1]
        seq_num = temp.rsplit('.', 1)[0]
        seq_num = int(seq_num) % 100000
        action = get_line(action_file, seq_num)
        file.write(action)
        file.write('\n')
    else:
        for action in get_random_action(peds_num, action_file):
            file.write(action)
            file.write('\n')
        file.close()
    return 1


if __name__ == '__main__':
    location_dict = retrieve_labeled_position(labeled_location_path)
    count = 0
    batch_adjust = 0

    '''import pickle
    white_list = []
    with open(r'C:\\Users\WEI_CHEN\Desktop\dynamoDb status success list.ob', 'rb') as f:
        white_list = pickle.load(f)
    print(white_list)'''

    # while count < ani_num:
    while count < 300:
        filename = 'vid_%08d.txt' % (start + count + batch_adjust * 100000)
        seq_name = 'seq_%08d' % (start + count + batch_adjust * 100000)
        # if seq_name in white_list:
        # count += 1
            # continue
        file = os.path.join(output_path, filename)
        a = create_scenario(file, location_file=r'C:\Users\12595\Desktop\mta-98\locations.txt',
                            peds_num=1, action_file=animation_path, mode='random_labeled')
        count += 1

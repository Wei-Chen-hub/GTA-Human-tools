import re
import json
import math
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from collections import defaultdict

# These three paths are from local pc, need update !
map_path = r'C:\Users\12595\Desktop\mta-98-0505\mta-98\GTAV-HD-MAP-satellite.jpg'
json_dir = r'C:\Users\12595\Downloads\gta-map-sgementation-project\train.json'
map_obj = r'C:\Users\12595\Desktop\map-gta5\gta-3d-map.obj'  # sys.argv[1]

location_types = ['location_city', 'location_coast', 'location_desert', 'location_forest', 'location_suburb']
location_colors = ['#7290f4', '#f8e71c', '#9b9b9b', '#61b850', '#c52866']
location_colors_rgb = [[114, 143, 243], [248, 231, 29], [155, 155, 155], [97, 184, 80], [196, 40, 103]]


def create_2d_segmentation(map_p, json_p, save_p, transparency):
    with open(json_p) as f:
        info = json.load(f)
        annotations = info['labels'][0]['annotations']

        img = Image.open(map_p)
        img2 = img.copy()
        draw = ImageDraw.Draw(img2)
        for locale_num in range(len(location_types)):
            for polygon in annotations[location_types[locale_num]]:
                coord = polygon['data']['points']
                x = [pt[0] * 8192 for pt in coord]
                y = [pt[1] * 8192 for pt in coord]
                xy = []
                for i in range(len(x)):
                    xy.append((x[i], y[i]))
                draw.polygon(xy, fill=location_colors[locale_num])

        img3 = Image.blend(img, img2, transparency)
        img3.save(save_p, quality=100)


def get_location_set(segmented_map_p):
    img = Image.open(segmented_map_p)
    im = np.array(img)

    '''by_color = defaultdict(int)
    for pixel in img.getdata():
        by_color[pixel] += 1
    remove_key_set = []
    for keys in by_color:
        if not 100000 <= by_color[keys]:
            remove_key_set.append(keys)
    for keys in remove_key_set:
        by_color.pop(keys)
    print(by_color)'''

    location_point_set = {}

    for locale_num in range(len(location_types)):
        temp = []
        color = (location_colors_rgb[locale_num][0], location_colors_rgb[locale_num][1], location_colors_rgb[locale_num][2])
        # color = location_colors[locale_num]
        x, y = np.where(np.all(im == color, axis=2))
        for a in range(len(x)):
            '''if location_types[locale_num] == 'location_desert':
                if not 2500 <= x[a] <= 4200 and 3500 <= y[a] <= 6700:
                    print(1)
                    continue'''
            temp.append([
                1.5229 * y[a] - 5720.4,
                -1.5186 * x[a] + 8390.3
            ])
        location_point_set[location_types[locale_num]] = temp
    print('Successfully retrieved location pt set!')
    return location_point_set


def get_a_location(l_set, l_type):
    if l_type is None:
        l_type = random.choice(location_types)
    else:
        l_type = [s for s in location_types if l_type in s]
        l_type = l_type[0]
    l_list = l_set[l_type]
    [x, y] = random.choice(l_list)
    x = x + random.uniform(-1, 1)
    y = y + random.uniform(-1, 1)

    # jpg_x = 0.6566 * float(map_x) + 3756
    # jpg_y = -0.6585 * float(map_y) + 5525
    # convert to gta world coord
    # print(map_x, map_y, l_type[0][9:])
    return x, y, l_type[9:]


class HashTable:
    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    # Insert values into hash map
    def set_val(self, key, val):

        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be inserted
            if record_key == key:
                found_key = True
                break

        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    # Return searched value with specific key
    def get_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key being searched
            if record_key == key:
                found_key = True
                break

        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    # Remove a value with specific key
    def delete_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be deleted
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    # To print the items of hash map
    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


def get_vertices(map_obj_p):
    reComp = re.compile("(?<=^)(v |vn |vt |f )(.*)(?=$)", re.MULTILINE)
    with open(map_obj_p) as f:
        data = [txt.group() for txt in reComp.finditer(f.read())]
    v_arr, vn_arr, vt_arr, f_arr = [], [], [], []
    for line in data:
        tokens = line.split(' ')
        if tokens[0] == 'v':
            v_arr.append([float(c) for c in tokens[1:]])
        elif tokens[0] == 'vn':
            vn_arr.append([float(c) for c in tokens[1:]])
        elif tokens[0] == 'vt':
            vn_arr.append([float(c) for c in tokens[1:]])
        elif tokens[0] == 'f':
            f_arr.append([[int(i) if len(i) else 0 for i in c.split('/')] for c in tokens[1:]])
    vertices, normals = [], []
    for face in f_arr:
        for tp in face:
            vertices += v_arr[tp[0] - 1]
            normals += vn_arr[tp[2] - 1]

    print(len(vertices))
    # Create a hash table to store vertices
    # vertices_table = HashTable(900000)
    xyz = []
    for i in range(int(len(vertices) / 3)):
        xyz.append([vertices[3 * i], vertices[3 * i + 2], vertices[3 * i + 1]])
    # xyz = unique_l(xyz)
    print(len(xyz))
    # vertices_table.set_val(key=(vertices[3 * i], vertices[3 * i + 2]), val=vertices[3 * i + 1])

    '''import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z,
                    cmap='viridis', edgecolor='none')
    plt.show()'''

    # print(vertices)
    return xyz


def get_z_from_xy(vertices, target_xy):
    vertices_xyz = vertices
    possible_vertices = get_possible_points(vertices_xyz, target_xy)
    distances = []
    for pts in possible_vertices:
        distances.append(math.sqrt((pts[0] - target_xy[0]) ** 2 + (pts[1] - target_xy[1]) ** 2))

    sorted_vertices = [ele for _, ele in sorted(zip(distances, possible_vertices))]

    print(sorted_vertices)

    z = None
    while not z:
        pop = False
        try:
            z = get_z_from_plane(sorted_vertices[0], sorted_vertices[1], sorted_vertices[2], target_xy)
        except ZeroDivisionError:
            pop = True
        if max(sorted_vertices[0]):
            pass

    return z


def get_possible_points(xyz, target):
    possible_set = []
    distance = 1
    while len(possible_set) <= 10:
        possible_set = []
        for points in xyz:
            if abs(target[0] - points[0]) <= distance and abs(target[1] - points[1]) <= distance:
                possible_set.append(points)
        possible_set = unique_l(possible_set)
        distance = 2 * distance
        if distance >= 1000:
            return None
    return possible_set


def get_z_from_plane(pt1, pt2, pt3, xy):
    x, y = xy
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    x3, y3, z3 = pt3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)

    print("equation of plane is ", a, "x +", b, "y +", c, "z +",  d, "= 0.")

    return - (a * x + b * y + d) / c


def unique_l(list_a):
    list_b = []
    for values in list_a:
        if values not in list_b:
            list_b.append(values)
    return list_b


if __name__ == "__main__":
    save_path = r'GTAV-MAP-segmented.jpg'
    # create_2d_segmentation(map_path, json_dir, save_p=save_path, transparency=1)
    location_set = get_location_set(segmented_map_p=save_path)
    for i in range(100):
        x, y, l_t = get_a_location(l_set=location_set, l_type=None)  # None stands for random type
        print(x, y, l_t)
    # x, y = -420, 2600
    # z = get_z_from_xy(vertices=get_vertices(map_obj_p=map_obj), target_xy=[x, y])


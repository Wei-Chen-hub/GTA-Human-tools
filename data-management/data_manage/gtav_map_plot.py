import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

map_path = r'C:\Users\WEI_CHEN\Desktop\mta-98-0505\mta-98\GTAV-HD-MAP-satellite.jpg'
store_path = r'C:\Users\WEI_CHEN\Desktop\GTAV-123.jpg'

gtav_map_pts = [[-3430, 698], [36, 7688], [2870, -1270]]
map_jpg_pts = [[1505, 4890], [3778, 462], [5642, 6360]]


def get_line(filepath, line_count):
    file = open(filepath, 'r').readlines()
    return file[int(line_count) - 1]


def plot_gta_map(mta_path, pt_radius=10, store=store_path, map=map_path):
    map_pts = []
    for folder in os.listdir(mta_path):
        print(folder)
        try:
            line = get_line(os.path.join(mta_path, folder, 'log.txt'), 6)

            map_pts.append([line.rsplit(' ', 3)[1], line.rsplit(' ', 3)[2]])
        except:
            pass
    print(map_pts)
    im = cv2.imread(map)
    for map_pt in map_pts:
        map_x = map_pt[0]
        map_y = map_pt[1]
        jpg_x = 0.6566 * float(map_x) + 3756
        jpg_y = -0.6585 * float(map_y) + 5525
        image = cv2.circle(im, (int(jpg_x), int(jpg_y)), radius=0, color=(0, 0, 255), thickness=pt_radius)
    cv2.imwrite(store, image)


if __name__ == "__main__":
    plot_gta_map(mta_path = r'D:\GTAV\MTA')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import os

faulty_image = r'D:\GTAV\mta-sample\image_sample\seq_00052358\00000007.jpeg'
good_image = r'D:\GTAV\mta-sample\image_sample\seq_00050955\00000014.jpeg'


def check_fullscreen(seq_folder_path=r'D:\GTAV\mta-sample\image_sample',
                     # def check_fullscreen(seq_folder_path=r'D:\GTAV\test',
                     store_path=r'C:\Users\it_admin\Desktop\mta-98-1026\mta-98\not_fullscreen2.txt'):
    pts = [[200, 0], [1000, 10], [1500, 20]]
    faulty_seq = []
    download_error_list = []

    for dir in os.listdir(seq_folder_path):
        print(1)
        faulty_image_count = 0
        for image in os.listdir(os.path.join(seq_folder_path, dir)):

            im = cv2.imread(os.path.join(seq_folder_path, dir, image))

            rgb45 = 0
            rgb255 = 0

            for pt in pts:
                jpg_x, jpg_y = (pt[0], pt[1])
                try:
                    [r, g, b] = [float(im[jpg_y, jpg_x, 0]), float(im[jpg_y, jpg_x, 1]), float(im[jpg_y, jpg_x, 2])]
                except TypeError:
                    print('problem occurs at ' + dir)
                    download_error_list.append(dir)
                    break
                if 50 >= 0.333 * (r + g + b) >= 40:
                    rgb45 += 1
                if 260 >= 0.333 * (r + g + b) >= 250:
                    rgb255 += 1

            if rgb45 >= 1 and rgb255 >= 2:
                faulty_image_count += 1

        if faulty_image_count >= 2:
            faulty_seq.append(dir)

    download_error_list = list(set(download_error_list))
    print(faulty_seq)
    print(download_error_list)

    with open(store_path, 'a') as f:
        f.write(str(faulty_seq))
        f.write('\n')
        f.write(str(download_error_list))


def check_fullscreen_single_seq(seq_folder_path):
    pts = [[200, 0], [1000, 10], [1500, 20]]

    faulty_image_count = 0
    checked = 0
    for image in os.listdir(seq_folder_path):
        if not image.endswith('.jpeg'):
            continue
        im = cv2.imread(os.path.join(seq_folder_path, image))

        rgb45 = 0
        rgb255 = 0

        for pt in pts:
            jpg_x, jpg_y = (pt[0], pt[1])
            [r, g, b] = [float(im[jpg_y, jpg_x, 0]), float(im[jpg_y, jpg_x, 1]), float(im[jpg_y, jpg_x, 2])]
            if 50 >= 0.333 * (r + g + b) >= 40:
                rgb45 += 1
            if 260 >= 0.333 * (r + g + b) >= 250:
                rgb255 += 1

            if rgb45 >= 1 and rgb255 >= 2:
                faulty_image_count += 1
        if faulty_image_count >= 2:
            return False
        checked += 1
        if checked >= 10:
            break
    return True


if __name__ == '__main__':
    check_fullscreen()

    '''pt_radius = 10
    image = cv2.imread(faulty_image)

    pts = [[200, 0], [1000, 10], [1500, 20]]

    for pt in pts:
        jpg_x, jpg_y = (pt[0], pt[1])
        rgb = [image[jpg_y, jpg_x, 0], image[jpg_y, jpg_x, 1], image[jpg_y, jpg_x, 2]]
        print(rgb)

        image = cv2.circle(image, (int(jpg_x), int(jpg_y)), radius=0, color=(0, 0, 255), thickness=pt_radius)
    store = r'D:\GTAV\mta-sample\image_sample\seq_00052358\write.jpeg'

    cv2.imwrite(store, image)'''

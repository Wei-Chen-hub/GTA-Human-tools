import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from data_manage.gen_pointcloud import generate_pt_cloud


def get_an_image(mta_p):
    seq_l = os.listdir(mta_p)
    # seq = random.choice(seq_l)
    seq = seq_l[1]
    print('Looking at', seq)
    image_p = os.path.join(mta_p, seq, '00000010.jpeg')
    depth_p = os.path.join(mta_p, seq, 'raws', 'depth_00000010.raw')
    return image_p, depth_p


def generate_mask_ds_1():
    mask_ds_2d = np.full((1080, 1920), False)
    mask_ds_1 = np.full(1920 * 1080, True)

    row_count, col_count = [0, 0]

    for row in mask_ds_2d:
        # print(len(row))
        col_count = 0
        for col in row:
            if row_count % 2 == 0 and col_count % 2 == 0:
                mask_ds_2d[row_count, col_count] = True
                pass
            else:
                mask_ds_1[row_count * 1920 + col_count] = False
            col_count += 1
        row_count += 1

    return mask_ds_1


def ds_a_image(image_p, depth_p, target_size=None):
    if target_size is None:
        target_size = [960, 540]
    temp, filename = os.path.split(image_p)
    filename = filename.rsplit('.', 1)[0]
    depth_path = os.path.join(temp, 'raws', 'depth_' + filename + '.raw')
    stencil_path = os.path.join(temp, 'raws', 'stencil_' + filename + '.raw')

    rows = 1080
    cols = 1920
    channels = 4
    # img_c = plt.imread(image_p)
    # plt.imshow(img_c)
    # plt.show()

    # depth_image = o3d.geometry.Image(depth_p)
    '''rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth_image),
        depth_trunc=5.0,
        convert_rgb_to_intensity=False)'''
    pcd, points, colors, mask_depth = generate_pt_cloud(image_p, visualize=False)
    mask_ds_1 = generate_mask_ds_1()
    # mask_ds_1 = np.full(1920 * 1080, True)
    mask = np.logical_and(mask_depth, mask_ds_1)

    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(r'C:\Users\it_admin\Desktop', 'sample.pcd'), pcd)


if __name__ == '__main__':
    mta_p = r'F:\MTA-multi-p\pending_upload_multi'
    image_p, depth_p = get_an_image(mta_p)
    ds_a_image(image_p, depth_p)

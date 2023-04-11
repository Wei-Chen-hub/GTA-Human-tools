import cv2  #OpenCV包
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def generate_pointcloud_from_depth(depth_image,cx,cy,fx,fy,scalingFactor=1.0):
    
    rows = depth_image.shape[0]
    cols = depth_image.shape[1]
    channels = depth_image.shape[2]
    points = np.zeros((rows*cols,3))

    for v in range(rows):
        for u in range(cols):
            Z = scalingFactor/depth_image[v,u]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            if Z == 0:
                continue
            points[v*cols + u] = [X,Y,Z]
    
    return points

def get_color(img,rows, cols):
    img_row = img.shape[0]
    img_col = img.shape[1]
    channels = img.shape[2]
    np_colors = np.zeros((rows*cols,3))
    for v in range(rows):
        for u in range(cols):
            np_colors[v*cols + u] = img[int((v/rows)*img_row), int((u/cols)*img_col)]/256.

    return np_colors

# 首先确定原图片的基本信息：数据格式，行数列数，通道数
rows=720#图像的行数
cols=1280#图像的列数
channels =1# 图像的通道数，灰度图为1

# 利用numpy的fromfile函数读取raw文件，并指定数据格式
depth = np.fromfile(r'D:\depth.raw', dtype='float32')

# 利用numpy中array的reshape函数将读取到的数据进行重新排列。
depth = depth.reshape(rows, cols, channels)
color = np.fromfile(r'D:\color.raw', dtype='uint8')
color = color.reshape(1080, 1920, 4)
color = color[:, :, :3]
# 展示图像
# cv2.imshow('1',img)
# # 如果是uint16的数据请先转成uint8。不然的话，显示会出现问题。
# cv2.waitKey()
# cv2.destroyAllWindows()

cam = o3d.camera.PinholeCameraIntrinsic()
intrinsic =  np.array([[ 1.15803376e+03,  0.00000000e+00,  9.60000000e+02],
[ 0.00000000e+00, -1.15803376e+03,  5.40000000e+02],
[ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

# intrinsic =  np.array([[ 1158,  0.00000000e+00,  960],
# [ 0.00000000e+00, 1158,  540],
# [ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

# intrinsic =  np.array([[ 772,  0.00000000e+00,  640],
# [ 0.00000000e+00, -772,  360],
# [ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

extrinsic = np.array(
    [[ 8.93371392e-01,  4.49318992e-01, -0.00000000e+00,  3.47233501e+02],
 [-1.00234072e-01,  1.99293273e-01 , 9.74800145e-01, -1.36555747e+02],
 [-4.37996218e-01 , 8.70858562e-01, -2.23079981e-01, -3.17328237e+02],
 [0,0,0,1]]
)
fx = intrinsic[0, 0]
fy = intrinsic[1, 1]
cx = intrinsic[0, 2]
cy = intrinsic[1, 2]
width = 1280
height = 720
cam.set_intrinsics(width, height, fx, fy, cx, cy)

# plt.imshow(depth)
# plt.savefig('depth.png',size=(1280,720))

xyz = generate_pointcloud_from_depth(depth, cx,cy, fx, fy)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
np_colors = get_color(color, height, width)
pcd.colors = o3d.utility.Vector3dVector(np_colors)
pcd.transform(extrinsic)
print(np.array(pcd.colors))
o3d.visualization.draw_geometries([pcd])


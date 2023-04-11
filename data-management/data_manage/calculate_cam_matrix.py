import math
import numpy as np

SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920
fov = 50

camx = -456.476
camy = 134.802
camz = 65.589

ptx = -459.192
pty = 140
ptz = 64.402

x = ptx - camx
y = pty - camy
z = ptz - camz

pi = 3.1415926

rx = -12.89 * pi / 180
ry = 0
rz = 26.70 * pi / 180

cx = math.cos(rx)
cy = math.cos(ry)
cz = math.cos(rz)
sx = math.sin(rx)
sy = math.sin(ry)
sz = math.sin(rz)
dx = cy * (sz * y + cz * x) - sy * z
dy = sx * (cy * z + sy * (sz * y + cz * x)) + cx * (cz * y - sz * x)
dz = cx * (cy * z + sy * (sz * y + cz * x)) - sx * (cz * y - sz * x)

fov_rad = fov * pi / 180
f = (SCREEN_HEIGHT / 2.0) * math.cos(fov_rad / 2.0) / math.sin(fov_rad / 2.0)
print(f)
print('dx= ', dx)
print('dy= ', dy)
print('dz= ', dz)

x2d = ((dx * (f / dy)) / SCREEN_WIDTH + 0.5) * SCREEN_WIDTH
y2d = (0.5 - (dz * (f / dy)) / SCREEN_HEIGHT) * SCREEN_HEIGHT

print('x2d= ', x2d)
print('y2d= ', y2d)

i_matrix = np.mat([[f, 0, 960],
                   [0, -f, 540],
                   [0, 0, 1]])
r_matrix = np.mat([[cy * cz, cy * sz, -sy],
                   [cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy],
                   [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy]])
pt_matrix = np.mat([[ptx],
                    [pty], [ptz], [1]])
cam_matrix = np.mat([[camx], [camy], [camz]])
pixel_matrix = np.mat([[1], [1], [1]])

temp = r_matrix * cam_matrix

e_matrix = np.mat([[r_matrix[0, 0], r_matrix[0, 1], r_matrix[0, 2], -temp[0, 0]],
                   [r_matrix[1, 0], r_matrix[1, 1], r_matrix[1, 2], -temp[1, 0]],
                   [r_matrix[2, 0], r_matrix[2, 1], r_matrix[2, 2], -temp[2, 0]]])
temp= e_matrix * pt_matrix
pixel_matrix = i_matrix * temp

print('intrinsic matrix: \n', i_matrix)
print('extrinsic matrix: \n', e_matrix)

print('2d point: \n', pixel_matrix/pixel_matrix[2, 0])
print(pixel_matrix[2, 0])

'''
f = open(r'D:\GTAV\1matrix.txt', 'a')
f.write(np.fromstring(i_matrix, dtype=float))
f.write("\n")
f.write(np.fromstring(e_matrix, dtype=float))
f.close

'''


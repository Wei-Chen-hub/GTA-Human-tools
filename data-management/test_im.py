import os
import glob
import pickle
import numpy as np
import shutil


from convert_data import convert_smpl
import open3d as o3d


sample_dir = r'D:\ptc'


dst = r'C:\Users\12595\Desktop\GTA-test-data\smplx\gta_human2\annotations_multiple_person'
mta_dir = r'E:\gtahuman2_multiple'

def parse_summary(pathname):
    with open(pathname, 'r') as f:
        content = f.read().splitlines()
    ped_idx_list = content[0][1:-1].split(',')
    aaa = []
    for item in ped_idx_list:
        aaa.append(int(item))
    return aaa


smplx_ps = glob.glob(os.path.join(r'C:\Users\12595\Desktop\GTA-test-data\smplx\smplx_fitting_test', '*.npz'))
# print(len(smplx_ps))
for smplx_p in smplx_ps:
    base = os.path.basename(smplx_p)
    seq_idx, ped_idx = base[4:12], base[13:19]
    # import pdb; pdb.set_trace()
    valid_list = parse_summary(os.path.join(mta_dir, 'seq_' + seq_idx, 'summary.txt'))

    # print(valid_list)
    if int(ped_idx) in valid_list:

        shutil.copyfile(smplx_p, os.path.join(dst, base))
        print(base + ' copied')

# print(smplx)

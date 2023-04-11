import os
import pickle

pkl_load_dir = r'C:\Users\it_admin\Desktop\mta-2021-1210\python\test\joint_angles_calculate'
pkl_load_pathname = os.path.join(pkl_load_dir, 'seq_00007425_99619.pkl')

with open(pkl_load_pathname, 'rb') as f:
    content = pickle.load(f, encoding='latin1')

keys = ['is_male', 'ped_action', 'fov', 'keypoints_2d', 'keypoints_3d', 'occ', 'self_occ', 'num_frames', 'bbox_xywh',
        'weather', 'daytime', 'location_tag', 'location', 'betas', 'global_orient', 'body_pose', 'vertices', 'transl']

# content = [content['num_frames']]

print(len(content['body_pose'][5]))



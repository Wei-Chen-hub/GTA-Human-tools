import os
import shutil
import json

import py7zr
import shutil
from datetime import datetime

# down sampling to 16 depth map per seq
# 600 seqs before zip: 171G, after zip:23G


def unzip_some_seq(log_p='status.json', target_p=r'D:\GTAV\MTA', seqs_f=r'\\DESKTOP-T8N0DEH\E_gtav\new-batch\mta_zip', num='all'):
    if num == 'all':
        num = len(os.listdir(seqs_f))
    with open(log_p, 'r') as f:
        status = json.load(f)

    finished_status = ['pending upload', 'rejected action failure', 'success']
    completed = 0
    for file in os.listdir(seqs_f):
        if file.endswith('.7z'):
            try:
                if status[file[:-3]]:
                    continue
            except KeyError:
                pass

            with py7zr.SevenZipFile(os.path.join(seqs_f, file), 'r') as archive:
                archive.extractall(path=target_p)
            completed += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('unpacking......' + str(completed) + '/' + str(num) + '......at ' + current_time)
        if completed >= num:
            print('Unpack success')
            return True
    return True


def down_sample_a_seq(seq_p):

    # first 5 frames not wanted
    img_count = len([x for x in os.listdir(seq_p) if x.endswith('.jpeg')]) - 5
    depth_keep = []
    for i in range(16):
        temp = int(img_count * i / 16) + 5
        name = 'depth_{:08d}.raw'.format(temp)
        depth_keep.append(name)

    for file in os.listdir(os.path.join(seq_p, 'raws')):
        if file not in depth_keep:
            os.remove(os.path.join(seq_p, 'raws', file))



if __name__ == '__main__':
    unzip_some_seq()
    # down_sample_a_seq(seq_p=r'D:\GTAV\MTA\seq_00000034')

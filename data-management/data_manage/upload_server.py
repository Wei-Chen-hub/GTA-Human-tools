import ftplib
import os
import re
import json
import random
import sys
import shutil
from datetime import datetime


user = 'weichen'
pwd = 'Cheeran202846.'
ip_sg1 = '10.51.0.103'
ip_sg2 = '10.51.3.34'


def upload_pending_gta_SG2(num=None, log_p='status.json', pending_p=r'D:\GTAV\pending_upload', dest="cache/share_data/datasets/Point2SMPL/GTA-Human", ip=ip_sg2,
                    user_name=user, password=pwd):

    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(dest)

    pattern = '^(seq_)[0-9]{8}'
    seqs = [seq for seq in os.listdir(pending_p) if re.fullmatch(pattern, seq)]

    uploaded = 0
    if not num:
        num = 999999
    with open(log_p, 'r') as f:
        status = json.load(f)
    for seq in seqs:
        if uploaded >= num:
            break
        if status[seq] == 'pending upload':
            upload_folder_r(sess=session, base=pending_p, path=seq)
            status[seq] = 'uploaded to SG2'
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(seq + ' uploaded to SG2 at ' + current_time)
            uploaded += 1

    with open(log_p, 'w') as f:
        json.dump(status, f)
    session.storbinary('STOR ' + 'status.json', open(log_p, 'rb'))
    print('Status updated, total ' + str(uploaded) + ' seqs uploaded.')


def upload_folder_r(sess, base, path):
    try:
        sess.mkd(path)
    # ignore "directory already exists"
    except ftplib.error_perm as e:
        if not e.args[0].startswith('550'):
            raise

    # path = os.path.join(path, folder)
    for name in os.listdir(base + '/' + path):
        localpath = os.path.join(base, path, name)
        if os.path.isfile(localpath):
            sess.storbinary('STOR ' + path + '/' + str(name), open(localpath, 'rb'))
        elif os.path.isdir(localpath):

            '''try:
                sess.mkd(path + '/' + name)
            # ignore "directory already exists"
            except ftplib.error_perm as e:
                if not e.args[0].startswith('550'):
                    raise'''

            upload_folder_r(sess, base, path + '/' + name)


if __name__ == '__main__':

    for i in range(10):
        upload_pending_gta_SG2(num=100)

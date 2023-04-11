import json
import multiprocessing
import os
import shutil

from check_full_screen import check_fullscreen_single_seq
from peds_analysis_new import get_seperate_tags, analysis_peds_csv
from sampling import down_sample_a_seq, unzip_some_seq


reject_tags = [-1, 2, 5, 6, 9, 10, 11]
finished_status = ['pending upload', 'uploaded to SG2', 'success']


def filter_data_for_upload(mta_p, target_p, delete_origin=False, log_p='status.json'):
    # get some data first
    unzip_some_seq(num=100)

    with open(log_p, 'r') as f:
        status = json.load(f)
    for d in os.listdir(mta_p):
        reject = False
        input_sequence_path = os.path.join(r'D:\GTAV\MTA', d)

        # check if ground mesh grabbed successfully
        scenario_path = os.path.join(input_sequence_path, 'vid_' + d[-8:] + '.txt')

        try:
            if status[d] in finished_status:
                continue
        except KeyError:
            pass

        with open(os.path.join(scenario_path), 'r') as f:
            lines = f.readlines()
            z = lines[4].rsplit(' ', 2)[2]
            if float(z) == 1.5:
                reject = True
                status[d] = 'rejected mesh not loaded'
                print(d + ' rejected: ground mesh not loaded')
        if reject:
            if delete_origin:
                shutil.rmtree(os.path.join(mta_p, d))
                continue

        # peds analysis
        ped_csv_path = os.path.join(input_sequence_path, 'peds.csv')
        tags_dict = get_seperate_tags(analysis_peds_csv(ped_csv_path, ped_csv_tag=1))
        for ped_id in tags_dict:
            ped_attributes = tags_dict[ped_id]
            if any(attribute in ped_attributes for attribute in reject_tags):
                reject = True
                if -1 in ped_attributes:
                    status[d] = 'rejected ped not loaded'
                    if 6 in ped_attributes:
                        status[d] = 'rejected camera underground'
                else:
                    status[d] = 'rejected action failure'
                print(d + ' rejected: bad tags found')
                if delete_origin:
                    shutil.rmtree(os.path.join(mta_p, d))
                break
        if reject:
            continue

        # check full screen
        if not check_fullscreen_single_seq(os.path.join(mta_p, d)):
            reject = True
            print(d + ' rejected: not fullscreen')
            status[d] = 'rejected not fullscreen'
            if delete_origin:
                shutil.rmtree(os.path.join(mta_p, d))
            continue

        if not reject:
            print(d + ' accepted!!!!!')
            down_sample_a_seq(os.path.join(mta_p, d))
            shutil.move(os.path.join(mta_p, d), os.path.join(target_p, d))
            status[d] = 'pending upload'

    with open(log_p, 'w') as f:
        json.dump(status, f)
    res = sum(x == 'pending upload' for x in status.values())
    print('Currently ' + str(res) + ' pending upload')


if __name__ == '__main__':
    '''filter_data_for_upload(delete_origin=True,
                           mta_p=r'D:\GTAV\MTA',
                           target_p=r'D:\GTAV\pending_upload')'''
    jobs = []
    n = 0
    '''while n <= 540:
        for i in range(16):
            p = multiprocessing.Process(target=filter_data_for_upload(delete_origin=True,
                                                                      mta_p=r'D:\GTAV\MTA',
                                                                      target_p=r'D:\GTAV\pending_upload'))
            jobs.append(p)
            p.start()

        n += 1'''

    with open(r'\\DESKTOP-T8N0DEH\E_gtav\MTA-new-10x\status.json', 'r') as f:
        status = json.load(f)

    from collections import Counter
    res = Counter(status.values())
    print(res)

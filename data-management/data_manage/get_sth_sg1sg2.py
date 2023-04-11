import ftplib
import os
import random
import sys
import shutil

user = 'weichen'
pwd = 'Cheeran202846.'
ip_sg1 = '10.51.0.103'
ip_sg2 = '10.51.3.34'


def get_csv_seq_SG2(src=None, path_name="/cache/share_data/datasets/GTA_Auto_Gen/dataset", ip=ip_sg2,
                    user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()
    print(len(names))
    # for name in names:
    # csvname = name + '_peds.csv'
    # path = os.path.join(r'D:/GTAV/csvs', csvname)
    # if name.endswith('.zip'):
    # continue
    # ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % name + '/peds.csv', callback=open(path, 'wb').write)


def get_selected_seq_sg2(seq_list, path_name="/cache/share_data/datasets/GTA_Auto_Gen/dataset", ip=ip_sg2,
                         user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()

    for seq_num in seq_list:
        seq = 'seq_' + '%08d' % seq_num
        path = os.path.join(r'D:/GTAV', 'mta-sample')
        downloadFiles(session, base=path_name, filename=seq, destination=path)
        # ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % seq, callback=open(path, 'wb').write)
        print(seq + ' downloaded')


def get_1k_seq_sg2(src=None, path_name="/cache/share_data/datasets/GTA_Auto_Gen", ip=ip_sg2,
                   user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()
    list = []
    i = 0
    for i in range(1000):
        seq = random.choice(names)

        i = i + 1
        if seq.endswith('.zip') or int(seq[-6]) > 0:
            continue
        list.append(seq)

    for seq in list:
        path = os.path.join(r'D:/GTAV', 'mta-sample')
        downloadFiles(session, base=path_name, filename=seq, destination=path)
        # ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % seq, callback=open(path, 'wb').write)
        print(seq + ' downloaded')


def get_pickle_sg2(pickle_list, path_name="/cache/share_data/datasets/GTA_Auto_Gen/annotations", ip=ip_sg2,
                   # share/caizhongang/to_wc/models/smpl
                   user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()
    path = r'D:/GTAV/mta-sample'

    '''for file in names:
        patha = os.path.join(path, file)
        ftplib.FTP.retrbinary(self=session, cmd='RETR ' + file, callback=open(patha, 'wb').write)

    return 0'''

    for i in pickle_list:
        seq_name = "seq_" + '%08d' % i
        pickle_name = seq_name + '.pkl'
        # path = os.path.join(r'D:/GTAV', 'mta-sample', seq_name, pickle_name)
        path = os.path.join(r'D:/GTAV', 'mta-sample', 'overlay', pickle_name)
        print(pickle_name + ' downloaded')

        ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % pickle_name, callback=open(path, 'wb').write)


def get_folder_sg2(path_name="/lustre/weichen/MTA", local_destination=r'D:\GTAV\visualization\pending', ip=ip_sg2,
                   # share/caizhongang/to_wc/models/smpl
                   user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()
    path = r'D:/GTAV/mta-sample'

    '''for file in names:
        patha = os.path.join(path, file)
        ftplib.FTP.retrbinary(self=session, cmd='RETR ' + file, callback=open(patha, 'wb').write)

    return 0'''
    for file in names:
        downloadFiles(session, base=path_name, filename=file, destination=local_destination)


def get_seq_and_anno_sg2(wanted, path_name="/lustre/weichen/MTA", local_destination=r'D:\GTAV\visualization\pending', ip=ip_sg2,
                   # share/caizhongang/to_wc/models/smpl
                   user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()
    path = r'D:/GTAV/mta-sample'

    '''for file in names:
        patha = os.path.join(path, file)
        ftplib.FTP.retrbinary(self=session, cmd='RETR ' + file, callback=open(patha, 'wb').write)

    return 0'''
    for seq_num in wanted:
        file = 'seq_{:08d}'.format(seq_num)
        pkl_name = file + '.pkl'
        downloadFiles(session, base=path_name, filename=file, destination=local_destination)
        ftplib.FTP.retrbinary(self=session, cmd='RETR ' + path_name + '/annotations/' + pkl_name, callback=open(pkl_name, 'wb').write)


def downloadFiles(session, base, filename, destination):
    try:
        path = base + '/' + filename
        session.cwd(path)
        # clone path to destination
        os.chdir(destination)
        os.mkdir(os.path.join(destination, filename))
        print(os.path.join(destination, filename) + " built")
    except OSError:
        # folder already exists at destination
        pass
    except ftplib.error_perm:
        print("error: could not change to " + path)
        sys.exit("ending session")

    # list children:
    filelist = session.nlst()

    for file in filelist:
        try:
            # this will check if file is folder:
            session.cwd(path + '/' + file)
            # if so, explore it:
            downloadFiles(session, base=base, filename=filename + '/' + file, destination=destination)
        except ftplib.error_perm:
            # not a folder with accessible content
            # download & return
            os.chdir(os.path.join(destination, filename))
            # possibly need a permission exception catch:
            session.retrbinary(cmd="RETR " + file, callback=open(os.path.join(destination, filename, file), "wb").write)
    return


def delete_except_list(pickle_list, dir_path):
    for file in os.listdir(dir_path):
        try:
            if int(file[-6:]) in pickle_list or int(file[-6]) > 0:
                continue
            else:
                try:
                    shutil.rmtree(os.path.join(dir_path, file))
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
        except ValueError:
            pass


def get_image_sample_sg2(selected_seq=None, path_name="/cache/share_data/datasets/GTA_Auto_Gen/dataset", ip=ip_sg2,
                         user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()

    path = os.path.join(r'D:/GTAV', 'mta-sample', 'image_sample')
    if selected_seq == 'all':
        selected_seq = names
    image_count = 0

    if type(selected_seq) == list:

        for item in selected_seq:
            name = item
            try:
                a = int(item)
                name = 'seq_' + '{:08d}'.format(a)
            except ValueError:
                pass
            if name.startswith('seq'):
                pass
            else:
                continue

            folder_path = os.path.join(path, name)
            shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)
            sample_set = random.sample(range(5, 15), 3)

            for frame in sample_set:
                image_p = os.path.join(folder_path, '{:08d}.jpeg'.format(frame))
                image_p_cloud = name + '/' + '{:08d}.jpeg'.format(frame)
                try:
                    ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % image_p_cloud,
                                          callback=open(image_p, 'wb').write)
                    image_count += 1
                except ftplib.error_perm:
                    pass
                print('Image: ' + '{:08d}.jpeg'.format(frame) + ' from ' + name + ' downloaded')

    print('Downloaded ' + str(image_count) + ' images from ' + str(len(selected_seq)) + ' sequences')


def get_scenario_sg2(selected_scenario=None, path_name="/cache/share_data/datasets/GTA_Auto_Gen/dataset", ip=ip_sg2,
                     user_name=user, password=pwd):
    session = ftplib.FTP(ip, user=user_name, passwd=password)
    session.cwd(path_name)
    names = session.nlst()

    path = os.path.join(r'D:/GTAV', 'mta-sample', 'scenarios')

    if selected_scenario == 'all':
        selected_scenario = names
    scenario_count = 0

    if type(selected_scenario) == list:
        for item in selected_scenario:
            name = item
            try:
                a = int(item)
                name = 'seq_' + '{:08d}'.format(a)
            except ValueError:
                pass
            try:
                seq_num = int(name[-8:])
            except ValueError:
                continue
            scenario_p = os.path.join(path, 'vid_{:08d}.txt'.format(seq_num))
            scenario_p_cloud = name + '/' + 'vid_{:08d}.txt'.format(seq_num)
            try:
                ftplib.FTP.retrbinary(self=session, cmd='RETR %s' % scenario_p_cloud,
                                      callback=open(scenario_p, 'wb').write)
                scenario_count += 1
            except ftplib.error_perm:
                print('Scenario: ' + 'vid_{:08d}.txt'.format(seq_num) + ' failed')
                pass
    print('Total ' + str(scenario_count) + ' scenarios downloaded')


if __name__ == '__main__':
    # get_csv_seq_SG2()
    # get_1k_seq_sg2()
    '''pickle_list = [261, 1190, 1459, 1891, 2590, 2774, 2775, 6177, 6184, 6306, 6955, 7060, 7062, 7282, 7483, 7534, 8708,
                   8822, 10002, 10503, 11636, 11981, 13073, 15044, 16527, 18662, 19753, 20821, 24438, 25268, 25463,
                   25679, 26812, 27316, 27366, 28140, 29573, 31922, 32234, 33104, 33572, 33655, 37051, 37494, 37858,
                   39395, 41777, 42337, 43864, 44279, 46811, 48342, 43834]'''
    '''pickle_list = [806, 969, 1637, 3589, 4216, 4981, 4987, 5152, 6189, 7507, 7557, 11423, 11603, 11699, 14100, 15426,
                   15560, 16206, 16579, 17176, 18100, 18174, 18308, 18455, 20765, 20814, 21027, 21087, 21176, 21551,
                   21908, 22160, 22237, 22878, 23005, 23727, 24569, 25614, 23476, 26279, 26359, 26896, 27390, 28869,
                   29155, 29227, 29455, 29581, 29762, 30844, 31678, 33353, 34277, 35586, 35689, 36701, 37172, 37722,
                   37863, 37870, 41834, 55706]'''
    pickle_list = [7483, 13073, 27316, 32234, 33104, 37172, 37870, 43834]
    # pickle_list = [37870, 43834]
    get_selected_seq_sg2(pickle_list)
    get_pickle_sg2(pickle_list)
    # delete_except_list(pickle_list, r'D:/GTAV/mta-sample')

    # get_image_sample_sg2(selected_seq=test_list)
    # get_folder_sg2()
    # get_seq_and_anno_sg2(wanted=pickle_list)
    # get_scenario_sg2(selected_scenario='all')

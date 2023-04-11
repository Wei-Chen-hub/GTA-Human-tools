import os
import subprocess
import py7zr
import shutil
from datetime import datetime
import multiprocessing
# about 1.5x speed using multiprocessing


# txt = '"C:\\Program Files\\7-Zip\\7z.exe" a -tzip %s %s -pSECRET'%(zipFileName,' '.join(FilesListBelow2GB))
# out = subprocess.check_output(txt, shell = True)


def get_scenario(scene_pending=r'\\DESKTOP-T8N0DEH\E_gtav\MTA-multi-p\pending_scenario'):
    scene_p = r'D:\GTAV\MTA-scenarios'
    pending_num = len([name for name in os.listdir(scene_p)])
    print('current queue: ' + str(pending_num))
    if pending_num < 200:

        for i in range(200 - pending_num):  # 5 files at a time
            folder = os.listdir(scene_pending)
            if folder:
                try:
                    file = folder[0]  # select the first file's name
                    curr_file = scene_pending + '\\' + file  # creates a string - full path to the file
                    shutil.move(curr_file, scene_p)  # move the files
                    folder.pop(0)
                except:
                    continue


def zip_and_move_single(target_p=r'\\DESKTOP-T8N0DEH\E_gtav\MTA-multi-p\mta_zip', delete_origin=False):
    seq_p = r'D:\GTAV\MTA'
    for file in os.listdir(seq_p):
        if check_seq_is_correct(seq_p=os.path.join(seq_p, file)):
            target7z = os.path.join(target_p, file + '.7z')
            with py7zr.SevenZipFile(target7z, 'w') as archive:
                archive.writeall(seq_p + r'\\' + file, file)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(target7z + ' created at ' + current_time)

            if delete_origin:
                shutil.rmtree(os.path.join(seq_p, file))


def check_seq_is_correct(seq_p):
    try:
        with open(os.path.join(seq_p, 'summary.txt'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('Done') and line.endswith('Done\n'):
                return True
    except:
        pass
    shutil.rmtree(seq_p)
    print(seq_p + ' failed and deleted')
    return False


def zip_and_move(delete_origin=False):
    jobs = []
    try:
        for i in range(8):
            p = multiprocessing.Process(target=zip_and_move_single(delete_origin=delete_origin))
            jobs.append(p)
            p.start()
    except:
        print('error occurred during threading')
        pass


if __name__ == '__main__':
    p = r'D:\GTAV\MTA'
    jobs = []
    for i in range(8):
        p = multiprocessing.Process(target=zip_and_move_single())
        jobs.append(p)
        p.start()
    # zip_and_move()
    # get_scenario()
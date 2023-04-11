import os
from subprocess import call

from angle_statistics import visualize_pose

RECORD_FPS = 15


def concat_video(overlay_dir, dst):
    for seq in os.listdir(overlay_dir):
        if seq.startswith('seq_'):
            pass
        else:
            continue
        img = os.path.join(overlay_dir, seq)

        cmd = (f'ffmpeg.exe -y -r {RECORD_FPS} -f image2 -s 1920x1080 -i {img}\\%8d.jpeg '
               f'-vcodec libx264 -crf 25 -pix_fmt yuv420p {dst}\\{seq}_smpl.mp4'
               )
        call(cmd, shell=True)


if __name__ == '__main__':
    ''' pickle_list = [806, 969, 1637, 3589, 4216, 4981, 4987, 5152, 6189, 7507, 7557, 11423, 11603, 11699, 14100, 15426,
      15560, 16206, 16579, 17176, 18100, 18174, 18308, 18455, 20765, 20814, 21027, 21087, 21176, 21551,
      21908, 22160, 22237, 22878, 23005, 23727, 24569, 25614, 23476, 26279, 26359, 26896, 27390, 28869,
      29155, 29227, 29455, 29581, 29762, 30844, 31678, 33353, 34277, 35586, 35689, 36701, 37172, 37722,
      37863, 37870, 41834, 55706]'''

    img_dir_path = r'C:\Users\it_admin\Desktop\GTA-test-data\mta'
    pkl_dir_path = r'C:\Users\it_admin\Desktop\GTA-test-data\smpl_fitting'
    overlay_out_path = r'C:\Users\it_admin\Desktop\GTA-test-data\smpl_vis'
    video_destination = r'C:\Users\it_admin\Desktop\GTA-test-data\vids'

    visualize_pose(img_load_dir=img_dir_path, pkl_load_dir=pkl_dir_path, img_save_dir=overlay_out_path)

    # concat_video(overlay_dir=overlay_out_path, dst=video_destination)

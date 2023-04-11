'''
简易自动启动游戏采集数据脚本
'''
import glob
import os
import time

import psutil

import admin
from get_seq_move_zip import zip_and_move, get_scenario
from key_input import press


def check_gta_process():
    plist = [p.name() for p in psutil.process_iter()]
    if 'GTA5.exe' in plist:
        return True
    else:
        return False


def gta_run():
    try:
        os.system('taskkill /f /im GTA5.exe')
    except:
        pass
    # ensure your steam already started
    # os.system('start steam://rungameid/271590')
    keydict = {'left': 0xCB, 'enter': 0x1C, 'F8': 0x42, 'right': 0xCD, 'esc': 0x01}
    if not admin.isUserAdmin():
        admin.runAsAdmin()
    get_scenario()
    while len(glob.glob('//DESKTOP-T8N0DEH//E_gtav//MTA-multi-p//pending_scenario//*')) > 0:
        press(keydict['F8'], 0)
        if not check_gta_process():
            os.system('D://GTAV//GTA5.exe --fullscreen')
            print('starting GTA V')
            while not check_gta_process():

                time.sleep(5)  # Wait the game start
            time.sleep(60)  # Wait for the Game start into the main menu

            press(keydict['right'], 5)
            press(keydict['enter'], 1)  # enter story mode
            time.sleep(90)  # Wait for entering the game
            press(keydict['F8'], 0)

        time.sleep(600)
        get_scenario()
        zip_and_move(delete_origin=True)


if __name__ == '__main__':
    gta_run()

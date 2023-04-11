from ftplib import FTP
import time
import os
import glob
import zipfile
import tqdm

current_path = os.path.dirname(os.path.abspath(__file__)) 
vid_path = 'D://Program Files (x86)//Steam//steamapps//common//Grand Theft Auto V//MTA-Scenarios'
game_path = 'E://' 

def ftpconnect(host,username,password):
    ftp = FTP()
    ftp.set_debuglevel(2)
    ftp.connect(host,21)
    ftp.login(username,password)
    return ftp

def downloadfile(ftp,remotepath,localpath):
    bufsize = 1024
    fp = open(localpath,'wb')
    ftp.retrbinary('RETR  '+ remotepath,fp.write,bufsize)
    # 接受服务器上文件并写入文本
    ftp.set_debuglevel(0) # 关闭调试
    fp.close() # 关闭文件

def uploadfile(ftp,remotepath,localpath):
    bufsize = 1024
    fp = open(localpath,'rb')
    ftp.storbinary('STOR '+ remotepath, fp,bufsize) # 上传文件
    ftp.set_debuglevel(0)
    fp.close()

def zip_data(datapath,dataname,outdir, start, end, gap=1):
    startdir = os.path.join(datapath, dataname)
    paths = [[dirpath, dirnames, filenames] for dirpath, dirnames, filenames in os.walk(startdir)][1:]

    for s in range(start, end, gap):
        f_path = paths[s-start:s-start+gap]
        zipname = os.path.join(outdir,'MTA%08d_%08d.zip'%(s,s+gap-1))
        f = zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED)

        with tqdm.trange(len(f_path)) as t:
            for p in t:
                dirpath = f_path[p][0]
                t.set_description('Compressing file (%s)' % dirpath)
                filenames = f_path[p][2]
                for filename in filenames:
                    f.write(os.path.join(dirpath,filename))
    f.close()

if __name__ == '__main__':
    start = 5100
    end = 6600
    gap = 100
    
    zip_data(game_path,'MTA', 'E://',start,end,gap)
    
    ftp = ftpconnect("10.51.0.103","linzhengyu","246810sf@")
    
    s_range = range(start,end,gap)
    for s in s_range:
        zipname = 'MTA%08d_%08d.zip'%(s,s+gap-1)
        zippath = 'E://'+zipname
        uploadfile(ftp,"/share/DSK/datasets/gta_raw/"+zipname,zippath)
    ftp.quit()
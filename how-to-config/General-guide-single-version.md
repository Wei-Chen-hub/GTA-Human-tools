## Installation

### VS Code 环境配置

1. 按如下方式配置环境变量，修改红框中的路径
2. 最后一项是将编译好的.asi拷贝到GTA.exe的同级目录下，可根据实际情况修改拷贝到的路径：
```
xcopy /Y $(SolutionDir)$(Platform)\$(Configuration)\$(ProjectName).asi "D:\Program Files\Epic Games\GTAV"
```
3. 配置完成后按F7进行编译，如果配置正确，则生成解决方案成功。
4. 注意：如果编译时GTA正在运行，则最后xcopy可能会失败，此时只需关闭GTA，再重新编译

![config_step1](how-to-config/img-instruction/Old-version1.png)
![config_step2](how-to-config/img-instruction/Old-version2.png)
![config_step3](how-to-config/img-instruction/Old-version3.png)
![config_step4](how-to-config/img-instruction/Old-version4.png)

### 运行采集
1. 将动作列表文件拷贝至GTA.exe的同级目录下，并重名为`PedAnimList.txt`，文件格式示例：
```
melee@unarmed@streamed_core walking_punch 2466.67
amb@code_human_wander_drinking@beer@male@idle_a idle_b 5033.33
amb@world_human_sit_ups@male@idle_a idle_c 15500
swimming@swim run 5333.34
```
为方便测试，可拷贝`PedAnimList_valid_timed_filter_example.txt`并重名为`PedAnimList.txt`

2. 生成config文件，将生成好的文件拷贝至GTA.exe同级目录下的`MTA-Scenarios`
```
python gen_config.py
```
其中每个文件代表一个数据采集的config，文件格式示例：
```
0 1e-08 15.571239061263284 3.6792001980034352 0 -13.29412878571644 0.0 180.0 0
-184 153 50 -184 153 50
-184 153 50 -184 153 50
1 -184 153 50 0 -1 1.0 -184 153 50 -184 153 50 1000000 0 29 1 1 1
```
每行对应的意义为：
```
(1) cCoords.x, cCoords.y, cCoords.z, stop, cRot.x, cRot.y, cRot.z, nAnims
(2) vTP1.x, vTP1.y, vTP1.z, vTP1_rot.x, vTP1_rot.y, vTP1_rot.z
(3) vTP2.x, vTP2.y, vTP2.z, vTP2_rot.x, vTP2_rot.y, vTP2_rot.z
(4) nPeds, pos.x, pos.y, pos.z, ngroup, currentBehaviour, speed, goFrom.x, goFrom.y, goFrom.z, goTo.x, goTo.y, goTo.z, task_time, type, radius, min_lenght, time_between_walks, spawning_radius
```
其中比较重要的参数是`nAnims`和`nPeds`，前者表示读取的动作编号，例如`nAnims=0`表示读取`PedAnimList.txt`的第一行的动作；后者表示行人的数量，`nPeds=1`表示画面中出现一个人。另外，`currentBehaviour`在这个采集脚本中固定为`-1`，在代码中会进入`spawn_peds_with_animation()`这个函数的逻辑。

3. 运行GTA游戏，进入故事模式后点击F8进行采集
4. 可视化采集的结果

```
python data_vis_demo.py
```
如果没有安装ffmpeg.exe，可以到[ffmpeg官网](https://www.gyan.dev/ffmpeg/builds/)或者点击[链接](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)下载，并将bin中的ffmpeg.exe拷贝到`data_vis_demo.py`的同级目录下。
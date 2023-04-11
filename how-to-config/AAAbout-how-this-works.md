### 运行采集 (Need 'DatasetAnnotator' compiled)
1. 将动作列表文件拷贝至GTA.exe的同级目录下，并重名为`PedAnimList.txt`，文件格式示例：
```
melee@unarmed@streamed_core walking_punch 2466.67
amb@code_human_wander_drinking@beer@male@idle_a idle_b 5033.33
amb@world_human_sit_ups@male@idle_a idle_c 15500
swimming@swim run 5333.34
```
为方便测试，可拷贝`PedAnimList_valid_timed_filter_example.txt`并重名为`PedAnimList.txt`

2. 生成config文件，将生成好的文件拷贝至GTA.exe同级目录下的`MTA-Scenarios`
See python part of generation of Scenario Files

3. 运行GTA游戏，进入故事模式后点击F8进行采集
Please wait
See python part for automation
4. 可视化采集的结果
See python part of data-vis & point cloud vis


如果没有安装ffmpeg.exe，可以到[ffmpeg官网](https://www.gyan.dev/ffmpeg/builds/)或者点击[链接](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)下载，并将bin中的ffmpeg.exe拷贝到`data_vis_demo.py`的同级目录下。
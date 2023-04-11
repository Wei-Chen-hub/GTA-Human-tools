# MTA Config Format

配置文件命名： `vid_%08d.txt`

配置文件内容格式

```
peds_number
center_point.x          .y                  .z
camera_displacement.x   .y                  .z
ped_action_name         ped_action_dict     ped_action_time    (for ped 1)
ped_action_name         ped_action_dict     ped_action_time    (for ped 2)
...

```

内容示例

```
1
2124.625732 4805.270020 40.479958 special
-3.024773113604502 -1.634296203754723 0.18765714359248453
-3.1242445401118126 0.0 -61.617481741677615
amb@incar@male@patrol@ds@idle_a idle_b 13233.3
```
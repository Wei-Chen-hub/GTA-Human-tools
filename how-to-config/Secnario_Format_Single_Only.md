# MTA Config Format

配置文件命名： `vid_%08d.txt`

配置文件内容格式

```
moving cx cy cz stop rx ry rz [anim_dict_index]
x y z x y z
x y z x y z
n1 x y z 0 behavior1 speed x y z x y z task_time type radious minimal_length time_between_walks spawning_radius
n2 x y z 0 behavior2 speed x y z x y z task_time type radious minimal_length time_between_walks spawning_radius

```

具体示例：

MTA Universal V0.4：vid_00000000.txt，内容示例

```
0 1e-8 8.8550 9.5754 0 -27.6858 0.0 180.0 2
683 -752 50 683 -752 50
32 683 -752 50 0 -1 1.0 683 -752 50 683 -752 50 1000000 0 15 1 1 1 
```

MTA Climb v1.1: vid_00000001.txt，内容示例
```
0 4.839069074162387 -3.7039068902095624 5.9332805314139865 0 -44.234916271592226 0.0 49.52499889264692
-872.162 177.222 69.0341 -872.162 177.222 69.0341
-872.162 177.222 69.0341 -872.162 177.222 69.0341
1 -872.162 177.222 69.0341 0 9 1.0 -872.162 177.222 69.0341 -873.989 177.572 71.3565 1000000 0 21 1 1 1

```


## Line 1:

​	camMoving: indicate if the camera moves or not during recording

​	if camMoving: 

​		A.x,A.y,A.z: the coordinate of point A saved in the menu

​		stop: true if the player is locked else false

​		B.x,B.y,B.z:  the coordinate of point B saved in the menu

​		C.x,C.y,C.z:  the coordinate of point C saved in the menu

​	else:

​		CamCoord: the coordinate of the camera

​		stop: true if the player is locked else false

​		camRot: 3d-vector of the rotation of camera

## Line2:

​	TP1: the coordinate of teleport 1

​	TP1_rot: 3d-vector of the rotation of teleport 1

## Line3:

​	TP2: the coordinate of teleport 2

​	TP2_rot: 3d-vector of the rotation of teleport 2

## Line n(n > 3):

for each spawn_peds call, a line containing following information is added to the end of the log file:

​	nPeds: number of pedestrians spawned

​	x,y,z: the coordinate of the spawn point

​	group: if pedestrians spawned are grouped

​	currentBehavior: the task type of the spawned pedestrians

	{
	-1:"MTA Universal",
	0:"SCENARIO",
	1:"STAND",
	2:"PHONE",
	3:"COWER",
	4:"WANDER",
	5:"CHAT",
	6:"COMBAT",
	7:"COVER",
	8:"MOVE",
	9:"CLIMB"
	}
​	speed: speed set in task sub_menu if task is MOVE else default

​	go from: 3d start point set in task sub_menu if task is MOVE else default

​	go to: 3d destination point set in task sub_menu if task is MOVE else default

​	task_time: not sure

​	type: scenario type if task is SCENARIO else default

​	radius: wander radius if task is WANDER else default

​	minimal length:  minimal wander length if task is WANDER else default(not checked)

​	time between walks: not sure

​	spawning radius: not sure
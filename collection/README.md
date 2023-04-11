# GTAHuman Data Collector

Written in C++ and Visual Studio 2019, retargeted to Visual Studio 2022.

## Step 0:
Extarct GTAV in your desired path.

## Step 1:
Open `MTAMods.sln` in Visual Studio, **DatasetAnnotator > Properties**

In **General > General Properties**, chenge as in the red box

![Properties1](/collection/instructions/insturct1.png)

## Step 2:

In **C/C++ > General > Additional Include Directiories**, change the path to following:
```
...\GTAHuman-tools\collection\DatasetAnnotator      # path to DatasetAnnotator
...\gtav\ScriptHookV_SDK_1.0.617.1a\inc             # path to GTAV Script Hook
...\GTAHuman-tools\collection\packages\eigen-3.3.9  # path to Eigen
```
![Properties2](/collection/instructions/insturct2.png)

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

## Step 3:
In **Linker > General > Additional Library Directories**, change the path to following:
```
...\gtav\ScriptHookV_SDK_1.0.617.1a\lib             # path to GTAV Script Hook lib
```
![Properties3](/collection/instructions/insturct3.png)

## Step 4 (Optional):
In **Build Events > Post-Build Event > Command Line**, change command to following:
```
xcopy /Y $(SolutionDir)$(Platform)\$(Configuration)\$(ProjectName).asi "...\gtav\"  # path to your GTA folder
```
![Properties4](/collection/instructions/insturct4.png)

## Step 5:
Build ! ! ! (With **Debug x64**, as other settings might fail in some cases)

In successful build, you should see a **DatasetAnnotator.asi** in your GTA folder

# At the Beginning

By downloading this tool or using GTA-Human Dataset, you agree on the following statement: "I declare that I will use the this tool for research and educational purposes only, since I am aware that commercial use is prohibited. I also undertake to purchase a copy of Grand Theft Auto V."

For setting the GAME Hook, please see [this tool](https://github.com/fabbrimatteo/JTA-Mods).

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

## Step 6 (For GTAV):
Assumed you have extracted GTAV zip (About 90G)

(If you don't have a previous game save)
Then copy save file from unzipped GTAV : 
```
...\GTAV\Goldberg SocialClub Emu Saves\GTA V\0F74F4C4\SGTA50000
...\GTAV\Goldberg SocialClub Emu Saves\GTA V\0F74F4C4\cfg.dat
# To
...\AppData\Roaming\Goldberg SocialClub Emu Saves\GTA V\0F74F4C4 
# Change ... to your user folder e.g. C:\Users\it_admin
```

## Step 7:
Game Setting: 1920x1080 (in other resolution depth map might not be captured), fullscreen (fullscreen image will be captured)
Run **GTAV.exe**, in game select **Story Mode**, after loading press **F8**.

The **DatasetAnnotator.asi** plugin will read scenario files and write results as below:
```
\GTAV\MTA-Scenarios
\GTAV\MTA
```
I have provided 50k+ single person scenarios in the zip file, enjoy!


## Notes:
1. Flie **...\GTAV\playGTAV.exe** is a hacking file for GTAV, sometimes it will be quaratined by Windows Defender, please restore it.

## Installation

### VS Code 环境配置

#### Step 1:
General
Choose identical config

![config_step1](how-to-config/img-instruction/DatasetAnnotator1.png)


#### Step 2:
C/C++   General
Additional Include: 
Line1 'DatasetAnnotator' folder under this project
Line2 'inc' folder under ScriptHookV SDK (GTA api)
Line3 eigen package (3.3.9 is included in '/packages/eigen-3.3.9')

![config_step2](how-to-config/img-instruction/DatasetAnnotator2.png)


#### Step 3:
Linker  General
'lib' folder under ScriptHookV SDK

![config_step3](how-to-config/img-instruction/DatasetAnnotator3.png)


#### Step 4:
Post-Build Event
Copy to GTAV, same dir as GTAV.exe

![config_step4](how-to-config/img-instruction/DatasetAnnotator4.png)

注意：如果编译时GTA正在运行，则最后xcopy可能会失败，此时只需关闭GTA，再重新编译
## 0. 指引

>每个人的环境都不一样，遇到问题可以查看maybe_problems.md文件或在Issues上反馈。
>
>持续更新中
>
>English README.md：to be updated


![alt text](pictures_videos/output.gif)  

本仓库强化学习部分基于：

[N3PO Locomoton](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)

另附titatit四足模式训练环境：

[TITATIT-Quadruped Mode](https://github.com/DDTRobot/titatit_rl)

**参考环境**

| Environment        | Brief info   |
| --------   | ----- | 
| 显卡| RTX 3060 |
| CUDA | CUDA12.5 |
| 训练环境 | isaacgym |
| sim2sim| Webots2023 |
| ROS | ROS2 Humble |
| 推理 | RTX 3060 / Jetson Orin NX on TITA + tensorRT|
| 虚拟环境 | anaconda |



### 本次开源包含有三部分  

#### Isaac Gym仿真训练  

![alt text](<pictures_videos/isaac_gym.gif>)
    
#### sim2sim仿真  
        
[tita_rl_sim2sim2real](https://github.com/DDTRobot/tita_rl_sim2sim2real)

![alt text](<pictures_videos/sim_webots.gif>)
#### sim2real实机部署

[tita_rl_sim2sim2real](https://github.com/DDTRobot/tita_rl_sim2sim2real)

![alt text](pictures_videos/real_robot.gif)

## 1. 环境搭建
>如您已有配好的RL环境，请直接跳至第3节开始训练

#### 1.1 安装NVIDIA显卡驱动

**方式1：使用ubuntu软件中心安装驱动**
>http://www.nvidia.cn/Download/index.aspx?lang=cn


**方式2：端中使用apt工具包安装**

添加 PPA 源：  
```markdown
    sudo add-apt-repository ppa:graphics-drivers/ppa  
``` 
为系统安装依赖项以构建内核模块： 
```bash 
sudo apt-get install dkms build-essential  
```  
安装NVIDIA驱动  
```bash 
sudo ubuntu-drivers autoinstall  
```
系统会自动安装推荐版本驱动，安装完重启系统  
```bash 
sudo reboot  
```

#### 1.2 安装anaconda  
[anaconda-installation](https://www.anaconda.com/download/success)  

#### 1.3 安装cuda
[cuda-toolkit-installation](https://developer.nvidia.com/cuda-toolkit-archive)

使用以下指令检查是否成功安装

```bash
nvidia-smi
```

#### 1.4. 安装tenssorrt  
我的cuda版本是12.0,所以我安装tensorrt8.6.0  
[tensorRT-installation](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

#### 1.5. 安装issacgym  
[isaacgym-installation](https://developer.nvidia.com/isaac-gym/download)  

## 2. 测试环境

>注意不要照抄指令
>
><your_env_name>为你的虚拟环境名
>
><your_path>为对应文件路径  


#### 2.1. conda配置虚拟环境
```bash
conda create -n <your_env_name> python=3.8
```
<your_env_name>为你的虚拟环境名该环境配置，在你的anaconda安装路径<your_path>/anaconda3/envs能找到<your_env_name>这个虚拟环境  
#### 2.2. 激活环境
```bash
conda activate <your_env_name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_path>/anaconda3/envs/<your_env_name>/lib
```
能在终端开头看到<your_env_name>,说明激活成功

#### 2.3. 测试conda和issacgym是否安装成功
安装以下包
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
进入isaacgym安装路径
```bash
cd 你的路径/isaacgym/python && pip install -e .  
```
测试
```bash
cd examples && python 1080_balls_of_solitude.py
```
看到一堆球落到地上表示安装成功，若没有参考第4步的解决方法

#### 2.4. 可能遇到的问题，“Isaac Gym”没有反应,运行以下两个指令有其它问题查看maybe_problems.md
```bash
sudo prime-select nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

#### 2.5. 退出conda环境
```bash
conda deactivate
```

## 3. 开始训练

#### 3.1. 从github上下载代码
```bash
git clone https://github.com/DDTRobot/tita_rl.git
cd tita_rl
```
#### 3.2. 激活conda环境   
```bash 
conda activate <your_env_name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:your path/anaconda3/envs/<your_env_name>/lib
```
#### 3.3. 运行训练程序
```bash
python train.py --task=Tita 
```
显存不够会非常卡，看到如下图片，表示程序正常执行，ctrl+c退出

![alt text](pictures_videos/image-1.png)
    
测试使用的是NVIDIA GeForce RTX 3060，打开图形界面的话，会非常卡，建议关闭图形界面
    
![alt text](pictures_videos/image-2.png)\
    
为了解决显存不足卡顿的问题，我们可以使用--headless参数，这样程序会以命令行的形式运行，不会打开图形界面，这样可以节省显存，提高运行速度

```bash
python train.py --task=Tita --headless
```

![alt text](pictures_videos/image-3.png)  
     
![alt text](pictures_videos/c7f9d78b-e6f7-46a5-b9cc-187ca142d9f5.jpeg)

## 4. 测试训练成果
#### 4.1. 查看训练成果
训练好的文件在tita_rl/logs下，例如model_10000.pt，将它拷贝到tita_rl主目录下，然后运行能看到
```bash
python simple_play.py --task=Tita
```
![alt text](<pictures_videos/isaac_gym.gif>)
#### 4.2. 将tita_rl主目录下的test.onnx推理转成model_gn.engine做sim2sim仿真
```bash
/usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=model_gn.engine
```
至此，iaacgym仿真和推理部分已经完成，接下来转到sim2sim和sim2real部分。  

sim2sim2real参考：[tita_rl_sim2sim2real](https://github.com/DDTRobot/tita_rl_sim2sim2real)

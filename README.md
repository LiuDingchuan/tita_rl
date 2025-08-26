## 0. 指引

>每个人的环境都不一样，遇到问题可以查看maybe_problems.md文件或在Issues上反馈。
>
>持续更新中
>
>English README.md：to be updated

本仓库强化学习部分基于：

[N3PO Locomoton](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git)


**参考环境**

| Environment        | Brief info   |
| --------   | ----- | 
| 显卡| RTX 4090 |
| CUDA | CUDA12.4 |
| 训练环境 | isaacgym |
| sim2sim| Gazebo/Webots2023/Mujoco |
| ROS | ROS2 Humble |
| 推理 | RTX 4090 / Jetson Orin NX + tensorRT 10.3|
| 虚拟环境 | anaconda |



### 开源模块包括 

#### Isaac Gym仿真训练  

![alt text](<pictures_videos/isaac_gym.gif>)
Press "F" to switch the perspective in the simulation interface
    
#### sim2sim仿真  
        
[tita_rl_sim2sim2real](https://github.com/LiuDingchuan/tita_rl_sim2sim2real)

![alt text](<pictures_videos/sim_gazebo.gif>)

#### sim2real实机部署

[tita_rl_sim2sim2real/](https://github.com/LiuDingchuan/tita_rl_sim2sim2real)
[tita_rl_locomotion/diablo_pluspro_sim2real](https://github.com/LiuDingchuan/tita_rl_locomotion)

![alt text](pictures_videos/sim2real.gif)

## 启动训练
```bash
python train.py --task=diablo_pluspro --headless
```

## 测试训练成果

训练好的文件在tita_rl/logs下，例如tita_rl/logs/diablo_pluspro/Jul23_18-32-24_recover_stair_height_5.8, 会根据训练时的当前日期-时间排列，如果不load_run直接simple_play的话，会自动调用最近一次的log的最后一个checkpoint；当然更推荐使用--load_run来指定加载pt文件的目录
```bash
python simple_play.py --task=diablo_pluspro --load_run=/home/hilabldc/tita_rl/logs/diablo_pluspro/Jul23_18-32-24_recover_stair_height_5.8 --checkpoint
```
![alt text](<pictures_videos/isaac_gym.gif>)
### 将tita_rl主目录下的test.onnx推理转成model_gn.engine做sim2sim仿真
```bash
/usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=model_gn.engine
```
至此，iaacgym仿真和推理部分已经完成，接下来转到sim2sim和sim2real部分。  

sim2sim2real参考：[tita_rl_sim2sim2real](https://github.com/LiuDingchuan/tita_rl_sim2sim2real)

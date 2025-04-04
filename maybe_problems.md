可能遇到的问题，“Isaac Gym”没有反应,运行以下两个指令
```
        sudo prime-select nvidia
        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```
# titi_rl

Python环境：python3.8
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
依赖：
```
conda install matplotlib
pip install opencv-python
```

### 报错1
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
```
解决：降低numpy版本
```
conda install numpy==1.23.5
```

### 报错2
```
 import tensorboard
ModuleNotFoundError: No module named 'tensorboard'
```
解决
```
pip install tensorboard
```
### 报错3
```
ModuleNotFoundError: No module named 'onnx'
```
```
pip install onnx
```

### 报错4
```
ModuleNotFoundError: No module named 'onnx'
```
```
pip uninstall setuptools
conda install setuptools==58.0.4
```

### 报错5

导出model_gn.engin进行sim2sim时，可能会出现“Uncaught exception detected: Unable to open library: libnvinfer_plugin.so.8 due to libcudnn.so.8”的报错。这是由于cudnn的版本问题，这里可以用
```
find / -name "libcudnn.so.8" 2>/dev/null
```
查找系统中有无该文件，笔者这里是查不到，但是却可以在conda环境下查到libcudnn.so.9
```
find / -name "libcudnn.so.9" 2>/dev/null
```
这里证实了笔者的判断：conda环境创建的时候，cudnn的版本和tensorrt不匹配。解决方法：
```
conda install -c conda-forge cudnn=8
```
这样就可解决了，并通过第一个命令查找，如果能在环境中找到，就说明解决成功了
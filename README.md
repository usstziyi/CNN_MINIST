# PyTorch MNIST CNN 实现与 MPS 加速

基于PyTorch的手写数字识别CNN模型，针对Apple Silicon设备优化了MPS硬件加速支持。

## 项目结构
```tree
├── data/                 # 数据集目录
├── mnist_cnn.py          # 命令行版CNN模型实现
├── mnist_cnn_mps.ipynb   # Jupyter Notebook版，含可视化和MPS加速
├── mnist_cnn_mps.py      # Python脚本版，含MPS加速
├── model/                # 模型权重保存目录
├── macos_mps_test.ipynb  # macOS MPS测试代码
└── .gitignore            # Git忽略规则文件
```

## 环境要求
- Python 3.x
- PyTorch 2.0+ (支持MPS加速)
- torchvision
- matplotlib
- Jupyter Notebook (可选，用于运行.ipynb文件)

## 快速开始
### 安装依赖
```bash
pip install torch torchvision matplotlib jupyter
```

### 运行命令行版本
```bash
python mnist_cnn.py
```

### 运行Notebook版本
```bash
jupyter notebook mnist_cnn_mps.ipynb
```

### 运行Python脚本版本
```bash
python mnist_cnn_mps.py
```

## 功能特点
- 两层卷积网络架构，包含池化和全连接层
- 支持MPS/CPU自动设备检测与切换
- 完整的训练/测试/模型保存流程
- 预测结果可视化功能
- 训练过程损失和准确率实时监控

## 模型保存
训练完成后，模型权重将自动保存至`model/mnist_cnn_model.pth`

## 许可证
[MIT](https://opensource.org/licenses/MIT)
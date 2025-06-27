# LeNet5 Pytorch

## 文件结构

``` bash
root
| - data
| - | - RBF vector 和 demo 用训练和测试图片
| - datasets
| - | - demo 用 MNIST 数据集 (csv 格式)
| - datasets.py
| - | - dataset 代码
| - models.py
| - | - LeNet5、损失函数、accuracy 函数代码
| - train.py
| - | - 训练代码
| - infer.py
| - | - 推理代码
| - utils.py
| - | - 数据预处理、日志、可视化代码
| - demo_train.ipynb
| - | - 训练 demo
| - demo_infer.ipynb
| - | - 推理 demo
```

## 模型训练

参考 `demo_train.ipynb`

## 模型推理

参考 
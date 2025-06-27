# LeNet5 Pytorch

复现1998年论文《Gradient-based learning applied to document recognition》，通过 Pytorch 训练 LeNet-5 进行手写识别。

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
| - | - 数据预处理、日志代码
| - demo_train.ipynb
| - | - 训练 demo
| - demo_infer.ipynb
| - | - 推理 demo
```

## 模型训练

参考 `demo_train.ipynb`

## 模型推理

参考 `demo_test.ipynb`

## 训练配置

| 参数 | 数值 |
|:----|:-----|
| optimizer | SGD |
| lr | 1e-4 |
| momentum | 0.9 |
| weight decay | 1e-4 |
| batch size | 1e-4 |
| num epoches | 120 |

## 测试结果

| Test Loss | Test Accuracy |
|:----|:-----|
| 3.9517 | 97.7% | 
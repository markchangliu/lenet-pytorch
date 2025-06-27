# 数据来源

完整数据来源 `https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/`

# 格式

`label, pix-11, pix-12, pix-13, ...`

# 转化为图片格式

``` python
from utils import mnist_cvs_to_imgfolder

csv_p = "..."
export_dir = "..."
mnist_csv_to_imgfolder(csv_p, export_dir)
```


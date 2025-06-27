import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_func_map(
    cat_ids: torch.Tensor,
    outputs: torch.Tensor
) -> torch.Tensor:
    """
    Args
    - `cat_ids`: `torch.Tensor`, shape `(b, )`
    - `outputs`: `torch.Tensor`, shape `(b, 10)`

    Returns
    - `loss`: `torch.Tensor`, shape `(1, )`
    """
    batch_size = outputs.shape[0]
    target_dist = outputs[range(batch_size), cat_ids]
    j = torch.as_tensor(0.1)
    dist_exp = torch.exp(-outputs)
    loss = target_dist + torch.log(torch.exp(-j) + torch.sum(dist_exp, dim = 1))
    loss = torch.sum(loss, dim = 0) / batch_size
    return loss

def loss_func_mse(
    cat_ids: torch.Tensor, 
    outputs: torch.Tensor
) -> torch.Tensor:
    """
    Args
    - `cat_ids`: `torch.Tensor`, shape `(b, )`
    - `outputs`: `torch.Tensor`, shape `(b, 10)`

    Returns
    - `loss`: `torch.Tensor`, shape `(1, )`
    """
    # 直接取正确类别的距离的平均值
    batch_size = outputs.shape[0]
    target_dist = outputs[torch.arange(batch_size), cat_ids]
    loss = torch.mean(target_dist)  # 最小化正确类别的平均距离
    return loss

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif hasattr(m, "weight") and hasattr(m, "bias"):
        # S2 and S4
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

@torch.no_grad()
def eval_func(
    cat_ids: torch.Tensor,
    outputs: torch.Tensor
) -> torch.Tensor:
    """
    Args
    - `cat_ids`: `torch.Tensor`, shape `(b, )`
    - `outputs`: `torch.Tensor`, shape `(b, 10)`

    Returns
    - `loss`: `torch.Tensor`, shape `(1, )`
    """
    vals, pred_cat_ids = torch.min(outputs, dim = 1)
    correctness = pred_cat_ids == cat_ids
    num_corrects = torch.sum(correctness)
    acc = num_corrects / len(cat_ids)
    return acc

class C1(nn.Module):
    """
    Arch
    - input = (b, 1, 32, 32)
    - output = (b, 6, 28, 28)
    - kernel = 5
    - stride = 1
    - pad = 0, computed based on `Hout = floor((Hin + 2P - K) / S) + 1`
    """
    def __init__(self) -> None:
        super(C1, self).__init__()
        self.conv = nn.Conv2d(1, 6, 5, 1, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y

class S2(nn.Module):
    """
    Arch
    - input = (b, 6, 28, 28)
    - output = (b, 6, 14, 14)
    - kernel = 2
    - stride = 2
    - pad = 0, computed based on `Hout = floor((Hin + 2P - K) / S) + 1`
    """
    def __init__(self, ) -> None:
        super(S2, self).__init__()
        self.weight = nn.Parameter(torch.ones(6))
        self.bias = nn.Parameter(torch.zeros(6))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(x, 2, 2, 0) * 4
        y = self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)
        return y
    
class C3(nn.Module):
    """
    Arch
    - input = (b, 6, 14, 14)
    - output = (b, 16, 10, 10)
    - kernel 5
    - stride 1
    - pad = 0, computed based on `Hout = floor((Hin + 2P - K) / S) + 1`

    Connection
    - 前6个C3图: 连接每3个连续的S2图(如 S2[1,2,3] -> C3[0], S2[2,3,4] -> C3[1], ..., S2[6,1,2] -> C3[5])
    - 接下来6个C3图: 连接每4个连续的S2图
    - 接下来3个C3图: 连接非连续的4个S2图
    - 最后1个C3图: 连接所有6个S2图
    """
    def __init__(self) -> None:
        super(C3, self).__init__()
        convs_1to6 = [nn.Conv2d(3, 1, 5, 1, 0) for i in range(6)]
        convs_7to12 = [nn.Conv2d(4, 1, 5, 1, 0) for i in range(6)] 
        convs_12to15 = [nn.Conv2d(4, 1, 5, 1, 0) for i in range(3)]
        convs_16 = [nn.Conv2d(6, 1, 5, 1, 0)]

        self.connection_table = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],  # 前6组 (3输入)
            [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],  # 中6组 (4输入)
            [0, 1, 3, 4], [1, 2, 4, 5], [2, 3, 5, 0],  # 后3组 (4输入)
            [0, 1, 2, 3, 4, 5]  # 最后1组 (6输入)
        ]

        self.convs = nn.ModuleList(convs_1to6 + convs_7to12 + convs_12to15 + convs_16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []

        for in_c, conv in zip(self.connection_table, self.convs):
            x_in = x[:, in_c, :, :]
            y = conv(x_in)
            ys.append(y)
        
        y = torch.concat(ys, dim = 1)

        return y

class S4(nn.Module):
    """
    Arch
    - input = (b, 16, 10, 10)
    - output = (b, 16, 5, 5)
    - kernel = 2
    - stride = 2
    - pad = 0, computed based on `Hout = floor((Hin + 2P - K) / S) + 1`
    """
    def __init__(self) -> None:
        super(S4, self).__init__()
        self.weight = nn.Parameter(torch.ones(16))
        self.bias = nn.Parameter(torch.zeros(16))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(x, 2, 2, 0) * 4
        y = self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)
        return y

class C5(nn.Module):
    """
    Arch
    - input = (b, 16, 5, 5)
    - output = (b, 120, 1, 1)
    - kernel = 5
    - stride = 1
    - pad = 0, computed based on `Hout = floor((Hin + 2P - K) / S) + 1`
    """
    def __init__(self) -> None:
        super(C5, self).__init__()
        self.conv = nn.Conv2d(16, 120, 5, 1, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y

class F6(nn.Module):
    """
    Arch
    - input = (b, 120 * 1 * 1)
    - output = (b, 84)
    """
    def __init__(self, *args, **kwargs):
        super(F6, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(120, 84)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        return y

class Output(nn.Module):
    """
    Arch
    - input = (b, 84)
    - output = (b, 10)
    - rbf_vectors = (10, 84)
    """
    def __init__(
        self, 
        rbf_vectors: torch.Tensor
    ) -> None:
        super(Output, self).__init__()
        self.register_buffer("rbf_vectors", rbf_vectors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, 1, 84) - self.rbf_vectors.view(-1, 10, 84)
        y = torch.sum(y ** 2, dim = 2)
        return y

class LeNet5(nn.Module):
    def __init__(self, rbf_vectors: torch.Tensor) -> None:
        assert rbf_vectors.shape == (10, 84)

        super(LeNet5, self).__init__()
        self.c1 = C1()
        self.s2 = S2()
        self.c3 = C3()
        self.s4 = S4()
        self.c5 = C5()
        self.f6 = F6()
        self.output_layer = Output(rbf_vectors)
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(b, 1, 32, 32)`

        Returns
        - `outputs`: `torch.Tensor`, shape `(b, 10)`
        """
        y = self.c1(imgs)
        y = self.s2(y)
        y = F.sigmoid(y)
        y = self.c3(y)
        y = self.s4(y)
        y = F.sigmoid(y)
        y = torch.flatten(self.c5(y), start_dim=1)
        y = self.f6(y)
        y = F.tanh(y * 2/3) * 1.7159
        outputs = self.output_layer(y)

        return outputs
    
# Model: DNN
```python
#!/usr/bin/env python3

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_layers=4, hidden_dim=1024):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```
## iter 1 hideen layers
hidden_layers=4, hidden_dim=2048 
The Valid loss of best model:0.8764328757921854

hidden_layers=3, hidden_dim=2048
The Valid loss of best model:0.8589290579160055

hidden_layers=2, hidden_dim=2048
The Valid loss of best model:0.8700498541196188

## iter 2
hidden_layers=4, hidden_dim=1024 
Valid loss: 0.8261549472808838

## iter 3 Feature selection
### 删除 id
```python
    remove_list = [0]
```
The Valid loss of best model:0.9194943308830261

### 删除 Mental Belif
```python
# feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.
        # feat_idx = list(range(1,raw_x_train.shape[1]))
        feat_idx = list(range(raw_x_train.shape[1]))
        remove_list = [0,38,39,46,51,
                       56,57,64,69,
                       74,75,82,87]
        feat_idx = [x for x in feat_idx if x not in remove_list]
        #without index
        # feat_idx = list(range(35,raw_x_train.shape[1])) #without index and states
```
The Valid loss of best model:0.8002835710843405

### 删除州信息
```python
feat_idx = list(range(35, raw_x_train.shape[1]))
remove_list = [0,38,39,46,51,
                56,57,64,69,
                74,75,82,87] 
```
The Valid loss of best model:0.8772939840952555

删除州信息的同时，添加 batchnorm
The Valid loss of best model:0.8751344879468282 --> test score: 14.72

把 batchnorm 放到 ReLU 前面 
The Valid loss of best model:0.8515465259552002 --> test score: 1.37115


## iter 4 dropout 层
对于一个 Regression 网络，已经很简单了，加了 dropout 层只会欠拟合。

dropout_p = 0
The Valid loss of best model:0.8515465259552002 --> test score: 1.37115

## iter 5 不使用 L2 Regularization
L2 正则化从模型参数的角度提升模型的泛化能力。


The Valid loss of best model:0.8002855976422628

1e-10
The Valid loss of best model:0.8519046107927958 --> test score: 1.37585

## 隐藏层维数
hidden_layers=2, hidden_dim=256
test score: 2.28

hidden_layers=2, hidden_dim=512
test score: 1.42

## 简单的模型
```python
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
```
The Valid loss of best model:0.8441318869590759
test score: 0.83966

加上 BatchNorm
The Valid loss of best model:0.9136309226353964






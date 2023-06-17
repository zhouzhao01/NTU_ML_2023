# Model 
Sequence lever classifier

将训练集切分为等长的 sequence_length = crop_length，送入 RNN model，直接计算整串序列的 label

```python
class RNN_sequence_level(nn.Module):
    def __init__(self, input_dim=39, output_dim=41, hidden_layers=1, hidden_dim=256,dropout=0.5):
        super(RNN_sequence_level, self).__init__()
        
        self.RNN = nn.GRU(input_size=39, hidden_size=hidden_dim,
                          num_layers=hidden_layers, batch_first=True, bidirectional=True,
                          dropout=dropout)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x):
        # x: b, l, MFCC-dim
        # print(x.size())
        out, h_n = self.RNN(x) #out: b, l, 2*hidden_size
        output = self.output(out) #output:  b, l, output_dim
        output = output.view(-1, 41) #output: *, output_dim

        return output
```

## iter 1
crop_length = 31



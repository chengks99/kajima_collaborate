import torch
import torch.nn as nn

class Conv_Unit(nn.Module):
    def __init__(self, in_nodes, out_nodes, kernel_size, stride, padding):
        super(Conv_Unit,self).__init__()
        self.conv = nn.Conv2d(in_nodes,out_nodes,kernel_size,stride,padding)
        self.batch_norm = nn.BatchNorm2d(out_nodes)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        y = self.conv(x)
        y = self.batch_norm(y)
        out = self.relu(y)
        return out


class VGG_M(nn.Module):
    def __init__(self, no_class):
        super(VGG_M, self).__init__()
        self.features = nn.Sequential(
                        nn.ReLU(),
                        Conv_Unit(1,96, kernel_size = (7,7), stride=(2,2), padding=(1,1)),
                        nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
                        Conv_Unit(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                        nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
                        Conv_Unit(256, 384, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1)),
                        Conv_Unit(384, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1)),
                        Conv_Unit(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1)),
                        nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2)),
                        
                        # kernel seize reduced from (9,1) to (7,1) in the following layer because 
                        # some inputs were smaller in dimension and resulted in crash with (9,1)
                        Conv_Unit(256, 4096, kernel_size = (7,1), stride=(1,1), padding = (0,0)), 
                        
                        nn.AdaptiveAvgPool2d((1,1)) 
                    )
        
        self.dense = nn.Sequential(
                        nn.Linear(4096,1024),
                        nn.ReLU(),
                        nn.Linear(1024, no_class)
                    )

    def forward(self,x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        # print(out.shape)
        return out
        
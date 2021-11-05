import torch
import torchvision
import torch.nn.functional as F

class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        original_model = torchvision.models.densenet121(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

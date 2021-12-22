import torch
import torchvision
import pytorch_pretrained_vit
import efficientnet_pytorch
import torch.nn.functional as F


def set_models(name, pretrained):
      
    if name == 'efficientnet_b4':  
        # model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')  
        model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b4') # no pretrained
        
        if pretrained:   ### use pretrained model

            # print("Load ImageNet pre-trained weights.")
            model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4') 


    elif name == "inception_v3": 
        model = torchvision.models.__dict__[name](pretrained=False, aux_logits=False)

        if pretrained:
            model = torchvision.models.__dict__[name](pretrained=True, aux_logits=False)

    
    elif name == "ViT": 
        model = pytorch_pretrained_vit.ViT('B_16', pretrained=False)

        if pretrained:
            model = pytorch_pretrained_vit.ViT('B_16', pretrained=True)

    else:
        model = torchvision.models.__dict__[name](pretrained=False)

        if pretrained:
            model = torchvision.models.__dict__[name](pretrained=True)


    return model





def set_out_features(model, outputs):  # alexnet, vgg19, inceptionv3, resnet101, densenet121, efficientnetB4, vit16
    """Changes number of outputs for the model.

    The change occurs in-place, but the new model is also returned."""

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        inputs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(inputs, outputs, bias=True)
        return model

        # including resnext101_32x8d
    elif (isinstance(model, torchvision.models.ResNet) or           
          isinstance(model, torchvision.models.Inception3)):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(inputs, outputs, bias=True)
        return model
        
    elif isinstance(model, torchvision.models.DenseNet):
        inputs = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputs, outputs, bias=True)
        return model

    elif isinstance(model, efficientnet_pytorch.EfficientNet):
        inputs = model._fc.in_features
        model._fc = torch.nn.Linear(in_features=inputs, out_features=outputs, bias=True)
        return model

    elif isinstance(model, pytorch_pretrained_vit.ViT):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features = inputs, out_features= outputs, bias=True)
        return model












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

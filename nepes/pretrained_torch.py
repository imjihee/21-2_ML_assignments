# 1. Directly Load a Pre-trained Model
# https://github.com/pytorch/vision/tree/master/torchvision/models
import torchvision.models as models
import torch.nn as nn
resnet50 = models.resnet50(pretrained=True) #load resnet50 model
# or model = models.resnet50(pretrained=False)

# Maybe you want to modify the last fc layer?
resnet50.fc = nn.Linear(2048, 2)  

pretrained_dict = resnet50.state_dict() 
model_dict = self_defined.state_dict() 
pretrained_dict = {k: v for k, v 
                    in pretrained_dict.items() if k in model_dict} 

# update & load
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)

# 3. Save & Load routines.
# routine 1
# torch.save(model.state_dict(), PATH)

# model = ModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

# routine 2
# torch.save(model, PATH)
# model = torch.load(PATH)
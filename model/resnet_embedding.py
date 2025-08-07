import torch
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.transforms = weights.transforms()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

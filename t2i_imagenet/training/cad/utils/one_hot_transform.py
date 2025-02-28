import torch
import torch.nn.functional as F
import torchvision


class OneHotTransform(torchvision.transforms.Lambda):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(
            torchvision.transforms.Lambda(self.one_hot_transform),
        )

    def one_hot_transform(self, x):
        x = torch.LongTensor([x])
        x = F.one_hot(x, num_classes=self.num_classes).squeeze(0).float()
        return x

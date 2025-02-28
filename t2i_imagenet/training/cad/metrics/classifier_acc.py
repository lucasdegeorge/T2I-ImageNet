import torch
import torchvision
from torchmetrics import Metric


class ClassifierAccuracy(Metric):
    def __init__(self, dataset="cifar"):
        super().__init__()
        if dataset == "CIFAR-10":
            from transformers import ViTForImageClassification

            self.num_classes = 10
            self.classifier_size = 224
            self.classifier = ViTForImageClassification.from_pretrained(
                "nateraw/vit-base-patch16-224-cifar10"
            )
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        (self.classifier_size, self.classifier_size), antialias=True
                    ),
                    torchvision.transforms.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                ]
            )
        elif dataset.startswith("ImageNet"):
            from transformers import DeiTForImageClassificationWithTeacher

            self.num_classes = 1000
            self.classifier_size = 384
            self.classifier = DeiTForImageClassificationWithTeacher.from_pretrained(
                "facebook/deit-base-distilled-patch16-384"
            )

            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        (438, 438),
                        antialias=True,
                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                    ),
                    torchvision.transforms.CenterCrop(self.classifier_size),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.244, 0.225],
                    ),
                ],
            )
        else:
            raise ValueError("Dataset not supported")

        self.classifier.eval()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, images, labels):
        # Remap images to [0, 1]
        images = images.float() / 255.0
        with torch.no_grad():
            images = self.transforms(images)
            output = self.classifier(images)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, torch.Tensor):
                logits = output
            else:
                raise ValueError("Output not supported")

            preds = torch.argmax(logits, dim=1)
            labels_true = torch.argmax(labels, dim=1)
            self.correct += (preds == labels_true).sum()
            self.total += len(labels)

    def compute(self):
        return self.correct / self.total

import clip
import torch
import torch.nn as nn
from PIL import Image


class MLP(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load(
            "checkpoints/sac+logos+ava1-l14-linearMSE.pth"
        )  # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)
        self.model.eval()
        self.model_clip, self.preprocess_clip = clip.load("ViT-L/14")

    def forward(self, images, device="cuda"):
        images = torch.stack(
            [self.preprocess_clip(image).to("cuda") for image in images]
        )
        with torch.no_grad():
            image_features = self.model_clip.encode_image(images)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(
            torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        )
        return prediction

import numpy as np
import pandas as pd
import streamlit
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('inference/best_my_resnet.pth', map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()
    return model


class WildImageDataset(Dataset):
    def __init__(self, imgs, transform: transforms.Compose = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = 0

        image = self.imgs[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


IN_SHAPE = (256, 256)

test_transforms = transforms.Compose([
    transforms.Resize(IN_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Model:
    def __init__(self):
        streamlit.header('')
        with streamlit.spinner('Loading ResNet...'):
            self.resnet = load_resnet()

    def make_dataloader(self, ids, text: list[str] | str, imgs: list[Image.Image]):
        dataset = WildImageDataset(imgs)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def predict(self, name: str, description: str, images: list[Image.Image]) -> np.ndarray:
        text = f"Название товара: {name}. Описание товара: {description}"
        dataloader = self.make_dataloader([i for i in range(len(images))], text, images)
        self.resnet.eval()

        all_probs = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(DEVICE)

                outputs = self.resnet(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_probs.extend(probs.flatten())

        return np.array(all_probs)
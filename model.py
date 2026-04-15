import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()
DEVICE = "mps" if torch.mps.is_available() else "cpu"

DATA_FOLDER = Path(r"data")
IMAGE_FOLDER = Path(r"data/images")


class ImageDataset(Dataset):
    def __init__(self, root_dir, df):
        self.samples = [str(root_dir) + "/" + str(i) + '.jpg' for i in df['id'].tolist()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[2], img.shape[1])
        min_side = min(w, h)
        img_cropped = transforms.CenterCrop(min_side)(img)
        img_resized = transforms.Resize((256, 256))(img_cropped)
        img_tensor = transforms.ToTensor()(img_resized)

        return img_tensor


class LoadedImageDataset(Dataset):
    def __init__(self, images):
        self.samples = images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        img = img.convert('RGB')
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[2], img.shape[1])
        min_side = min(w, h)
        img_cropped = transforms.CenterCrop(min_side)(img)
        img_resized = transforms.Resize((256, 256))(img_cropped)
        img_tensor = transforms.ToTensor()(img_resized)

        return img_tensor

class Model:
    def __init__(self):
        self.res_net50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
        train_df = pd.read_csv(DATA_FOLDER / "train.csv")
        tr_images_ds = ImageDataset(IMAGE_FOLDER, train_df)

        self.batch_size = 32

        images = DataLoader(tr_images_ds, batch_size=self.batch_size, shuffle=False)

        self.sureness50 = []

        for batch in tqdm(images):
            with torch.no_grad():
                image_sure50 = self.res_net50(batch.to(DEVICE))

            sure50 = image_sure50.to('cpu').numpy()

            self.sureness50.extend(sure50)
        # %%
        for i in range(1000):
            train_df[f"f{i}"] = [self.sureness50[j][i] for j in range(len(self.sureness50))]

        self.features = [f'f{u}' for u in range(1000)]
        target = 'label'
        # %%
        X_tr, X_val, y_tr, y_val = train_test_split(train_df[self.features], train_df[target], test_size=0.3, shuffle=True)
        # %%
        self.logreg_model = LogisticRegression(n_jobs=-1, solver='newton-cholesky')

        self.logreg_model.fit(X_tr, y_tr)
        pred = self.logreg_model.predict(X_val)
        print("Точность на валидационной выборке:", accuracy_score(pred, y_val))
        # %%
        self.logreg_model.fit(train_df[self.features], train_df[target])
        pass

    def predict(self, name : str, description : str, images : list[Image.Image]) -> np.ndarray:
        ts_images_ds = LoadedImageDataset(images)
        images_ts = DataLoader(ts_images_ds, batch_size=self.batch_size, shuffle=False)
        sureness50_ts = []

        for images_ in tqdm(images_ts):
            with torch.no_grad():
                sure = self.res_net50(images_.to(DEVICE))

            sureness50_ts.extend(sure.to('cpu').numpy().tolist())
        test_df = pd.DataFrame()
        for i in range(1000):
            test_df[f"f{i}"] = [sureness50_ts[j][i] for j in range(len(sureness50_ts))]

        pred_ts = self.logreg_model.predict_proba(test_df[self.features])
        return np.array([i[1] for i in pred_ts.tolist()])
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit
import torch
from PIL import Image
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER = Path(r"../data")
IMAGE_FOLDER = Path(r"../data/images")

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
        self.batch_size = 32
        self.img_emb_len = 512
        self.text_emb_len = 512
        self.features = [f'f{i}' for i in range(1000)]
        self.features += [f'img{i}' for i in range(self.img_emb_len)]
        self.features += [f'text{i}' for i in range(self.text_emb_len)]
        self.features.append('description')
        streamlit.header("")
        with streamlit.spinner('Loading ResNet...'):
            self.res_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(DEVICE)
        with streamlit.spinner('Loading CatBoost...'):
            self.catboost_model = CatBoostClassifier().load_model("ML_solve/Semøn/notebooks/catboost_model.cbm")
        with streamlit.spinner('Loading SentenceTransformer...'):
            self.emb_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                trust_remote_code=True,
                device=DEVICE,
                truncate_dim=512
            )

    def get_embeddings(self, texts, imgs):
        batch_size = 256

        text_embeds = []
        image_embeds = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            text_batch = texts[i: i + batch_size]
            img_batch = imgs[i: i + batch_size]
            img_paths = []
            for i in range(len(img_batch)):
                path = f"/tmp/ml_img{i}.jpg"
                img_paths.append(path)
                img_batch[i].save(path)

            text_emb = self.emb_model.encode(
                text_batch,
                normalize_embeddings=True,
                batch_size=len(text_batch),
                show_progress_bar=False,
                device=DEVICE
            )

            image_emb = self.emb_model.encode(
                img_paths,
                normalize_embeddings=True,
                batch_size=len(img_batch),
                show_progress_bar=False,
                device=DEVICE
            )

            text_embeds.extend(text_emb)
            image_embeds.extend(image_emb)

        return text_embeds, image_embeds

    def predict(self, name : str, description : str, images : list[Image.Image]) -> np.ndarray:
        text = f"Название товара: {name}. Описание товара: {description}"
        ts_images_ds = LoadedImageDataset(images)
        images_ts = DataLoader(ts_images_ds, batch_size=self.batch_size, shuffle=False)
        sureness_ts = []
        test_df = pd.DataFrame()
        test_text_emb, test_img_emb = self.get_embeddings([text] * len(images), images)

        for images_ in tqdm(images_ts):
            with torch.no_grad():
                sure = self.res_net(images_.to(DEVICE))

            sureness_ts.extend(sure.to('cpu').numpy().tolist())

        for i in range(1000):
            test_df[f"f{i}"] = [sureness_ts[j][i] for j in range(len(sureness_ts))]

        for i in range(self.text_emb_len):
            test_df[f"text{i}"] = [test_text_emb[j][i] for j in range(len(test_text_emb))]

        for i in range(self.img_emb_len):
            test_df[f"img{i}"] = [test_img_emb[j][i] for j in range(len(test_img_emb))]

        test_df['description'] = description

        pred_ts = self.catboost_model.predict_proba(test_df[self.features])
        test_df['y_pred'] = [i[1] for i in pred_ts.tolist()]
        return np.array([i[1] for i in pred_ts.tolist()])
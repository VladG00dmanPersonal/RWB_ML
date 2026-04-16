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

def load_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('inference/best_my_resnet.pth', map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()
    return model

def resnet_embeddings(loader, resnet):
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader):
            # если dataset возвращает (img, label)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(DEVICE)

            emb = resnet(images)  # (B, 2048)
            emb = emb.flatten(1)

            # нормализация (очень желательно)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

def load_sentence_transformer():
    return SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        trust_remote_code=True,
        device=DEVICE,
        truncate_dim=512
    )

def get_embeddings(df_test: pd.DataFrame, imgs: list[Image.Image], model):
    texts = df_test['text'].to_list()
    batch_size = 256

    text_embeds = []
    image_embeds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        text_batch = texts[i: i + batch_size]
        img_batch = imgs[i: i + batch_size]
        img_paths = []
        for j in range(len(img_batch)):
            img_batch[j].save(f"/tmp/ml_img{i}.jpg")
            img_paths.append(f"/tmp/ml_img{i}.jpg")

        text_emb = model.encode(
            text_batch,
            normalize_embeddings=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            device=DEVICE
        )

        image_emb = model.encode(
            img_paths,
            normalize_embeddings=True,
            batch_size=len(img_batch),
            show_progress_bar=False,
            device=DEVICE
        )

        text_embeds.extend(text_emb)
        image_embeds.extend(image_emb)
    return text_embeds, image_embeds

def get_emb_sim_and_l2(text_emb, img_emb):
    batch_size = 256
    test_similarities = []
    test_l2 = []

    for i in tqdm(range(0, len(text_emb), batch_size), desc="Test Batches"):
        text_embeds = np.array(text_emb[i: i + batch_size])
        image_embeds = np.array(img_emb[i: i + batch_size])

        text_embeds = torch.tensor(text_embeds).to(DEVICE)
        image_embeds = torch.tensor(image_embeds).to(DEVICE)

        cos_sims = util.cos_sim(text_embeds, image_embeds)
        diag_sims = torch.diag(cos_sims).cpu().numpy()
        test_similarities.extend(diag_sims)

        l2_dist = torch.norm(text_embeds - image_embeds, dim=1).cpu().numpy()
        test_l2.extend(l2_dist)

    return test_similarities, test_l2

class WildDataset(Dataset):
    def __init__(self, df: pd.DataFrame, has_labels=True):
        self.df = df
        self.has_labels = has_labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # text_emb = np.array(self.df.iloc[idx]['text_emb'], dtype=np.float32)
        # img_emb = np.array(self.df.iloc[idx]['img_emb'], dtype=np.float32)
        rn_img_emb = np.array(self.df.iloc[idx]['rn_img_emb'], dtype=np.float32)
        emb_sim = np.array([self.df.iloc[idx]['emb_sim']], dtype=np.float32)
        emb_l2 = np.array([self.df.iloc[idx]['emb_l2']], dtype=np.float32)
        # ti_emb = text_emb * img_emb

        combined = np.concatenate([rn_img_emb, emb_sim, emb_l2])
        combined = torch.from_numpy(combined)

        if self.has_labels:
            label = self.df.iloc[idx]['label']
            return combined, label
        else:
            return combined, 0


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

class VladBinaryNet(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims=None,
            dropout=0.3
    ):
        super(VladBinaryNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(
            nn.Linear(prev_dim, 1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze()


class Model:
    def __init__(self):
        streamlit.header('')
        #with streamlit.spinner('Loading SentenceTransformer...'):
        #    self.sentence_transformer = load_sentence_transformer()
        with streamlit.spinner('Loading ResNet...'):
            self.resnet = load_resnet()
        #with streamlit.spinner('Loading VladNet...'):
        #    self.vlad_net = VladBinaryNet(2050).to(DEVICE)

    def make_df(self, ids, text: list[str] | str, imgs: list[Image.Image]):
        df = pd.DataFrame({'id': ids})
        df['text'] = text
        df['text_emb'], df['img_emb'] = get_embeddings(df, imgs, self.sentence_transformer)
        dataset = LoadedImageDataset(imgs)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        df['rn_img_emb'] = resnet_embeddings(dataloader, self.resnet)
        df['emb_sim'], df['emb_l2'] = get_emb_sim_and_l2(df['text_emb'], df['img_emb'])
        return df

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
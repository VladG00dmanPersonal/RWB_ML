# %% [markdown]
# ## Dependencies

# %%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as Dst
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import torchvision.transforms as transforms
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from sentence_transformers import SentenceTransformer, util, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.base.evaluation import BaseEvaluator

from transformers import AutoModel

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from PIL import Image, ImageStat, ImageFilter
import cv2

from joblib import Parallel, delayed

from tqdm.cli import tqdm
from tqdm import trange

import random
import os

tqdm.pandas()

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_all_seeds(2)

# %%
DATA_PATH = "../data"
absolute_images_path = "~/home/vladg00dman/Projects/Hakatons/RWB_ML/ML_solve/Vlad/data/images"

df_path = f"{DATA_PATH}/train.csv"
df_test_path = f"{DATA_PATH}/test.csv"
df_sample_subm_path = f"{DATA_PATH}/sample_submission.csv"
images_path = f"{DATA_PATH}/images"

# %%
df = pd.read_csv(df_path)
df_test = pd.read_csv(df_test_path)
df_sample_subm = pd.read_csv(df_sample_subm_path)

# %%
def get_image_path(path, img_name):
    return f"{path}/{img_name}.jpg"

df['img_path'] = df['id'].apply(lambda x: get_image_path(images_path, x))
df['img_abs_path'] = df['id'].apply(lambda x: get_image_path(absolute_images_path, x))
df_test['img_path'] = df_test['id'].apply(lambda x: get_image_path(images_path, x))
df_test['img_abs_path'] = df_test['id'].apply(lambda x: get_image_path(absolute_images_path, x))

df.head()

# %% [markdown]
# ## EDA

# %%
df.head()

# %%
df.shape

# %%
df_test.shape

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %%
df['label'].value_counts()

# %%
plt.pie(
    x=df['label'].value_counts(),
    labels=['1', '0'],
)
plt.show()

# %%
def get_image(path: str) -> Image.Image:
    return Image.open(path)

df['img'] = df['img_path'].progress_apply(get_image)
df_test['img'] = df_test['img_path'].progress_apply(get_image)
df.head()

# %%
df.isna().sum()

# %%
df_test.isna().sum()

# %% [markdown]
# ### Features

# %%
def combine_texts(row):
    return f"Название товара: {row['name']}. Описание товара: {row['description']}"

df['text'] = df.apply(combine_texts, axis=1)
df_test['text'] = df_test.apply(combine_texts, axis=1)

# %%
def extract_image_features(pil_image: Image.Image, image_path: str = None) -> dict:
    """
    Возвращает ~25 полезных фич из PIL.Image для задачи релевантности фото-товара
    """
    features = {}
    
    # 1. Геометрия
    # w, h = pil_image.size
    # features['width'] = w
    # features['height'] = h
    # features['aspect_ratio'] = w / h if h != 0 else 0.0
    # features['total_pixels'] = w * h
    # features['is_square'] = int(abs(w - h) < 20)
    
    # # 2. Формат
    # features['mode'] = pil_image.mode
    # features['format'] = pil_image.format or 'unknown'
    # features['has_alpha'] = int(pil_image.mode in ('RGBA', 'LA', 'P'))
    
    # 3. Статистика цвета и яркости (через numpy — надёжно)
    # if pil_image.mode in ('RGB', 'RGBA'):
    #     img_array = np.array(pil_image.convert('RGB'), dtype=np.float32)
        
    #     mean_rgb = np.mean(img_array, axis=(0, 1))
    #     std_rgb = np.std(img_array, axis=(0, 1))
        
    #     r, g, b = mean_rgb
    #     features['brightness_mean'] = float(np.mean(mean_rgb))
    #     features['brightness_std'] = float(np.mean(std_rgb))
    #     features['contrast'] = float(np.mean(std_rgb))          # хороший прокси контраста
        
    #     # Цветность (насыщенность)
    #     features['colorfulness'] = float(
    #         np.sqrt(np.var(mean_rgb) + 
    #                 ((r - g)**2 + (r - b)**2 + (g - b)**2) / 2) / 100.0
    #     )
    # else:
    #     # grayscale fallback
    #     img_array = np.array(pil_image.convert('L'), dtype=np.float32)
    #     features['brightness_mean'] = float(np.mean(img_array))
    #     features['brightness_std'] = float(np.std(img_array))
    #     features['contrast'] = features['brightness_std']
    #     features['colorfulness'] = 0.0
    
    # 4. Резкость / размытость (Laplacian) — одна из самых полезных фич!
    img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['sharpness_laplacian'] = float(lap_var)
    features['is_blurry'] = int(lap_var < 120)          # порог можно подкрутить под твои данные
    
    # 5. Энтропия (правильно)
    # gray_pil = pil_image.convert('L')
    # hist = np.array(gray_pil.histogram(), dtype=float)
    # hist /= (hist.sum() + 1e-10)
    # hist = hist[hist > 0]
    # features['entropy'] = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
    
    # # 6. Гистограмма яркости — 16 бинов (не 256!)
    # hist_16 = np.array(gray_pil.histogram(), dtype=float).reshape(16, 16).sum(axis=1)
    # hist_16 /= (hist_16.sum() + 1e-10)
    # for i, val in enumerate(hist_16):
    #     features[f'hist_bin_{i}'] = float(val)
    
    # 7. Размер файла (если есть путь)
    if image_path and isinstance(image_path, str):
        try:
            features['file_size_kb'] = round(len(open(image_path, 'rb').read()) / 1024, 2)
        except:
            features['file_size_kb'] = 0.0
    
    return features

# %% [markdown]
# Размеры, мод, разрешение у картинок - всё одинаковое, поэтому не добавляем в фичи

# %%
# def process_row(row):
#     # Здесь находится ваша ресурсоемкая логика
#     # Например, сложные вычисления или обработка текста
#     result = extract_image_features(row['img'], row.get('img_path')) # Ваша логика
#     return result

# # Параллельное выполнение
# # n_jobs=-1 означает использование всех доступных ядер процессора
# results = Parallel(n_jobs=-1, verbose=100)(delayed(process_row)(row) for _, row in df.iterrows())
# results

# # Добавляем результаты обратно в DataFrame
# # df['new_column'] = results

# # Предполагаем, что в df есть колонка 'img' (PIL Image) и 'img_path'
# # feat_series = df.progress_apply(
# #     lambda row: extract_image_features(row['img'], row.get('img_path')), 
# #     axis=1
# # )

# # feat_df = pd.DataFrame(feat_series.tolist())
# # feat_df.head()

# %%
df.head()

# %% [markdown]
# ### Examples

# %%
def get_item(ind: int) -> None:
    print(df.iloc[ind])
    display(df.iloc[ind]['img'])

get_item(random.randint(0, df.shape[0] - 1))

# %%
ids = df[df['card_identifier_id'] == 1519].index.to_list()
for id in ids:
    get_item(id)

# %% [markdown]
# ## ML

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE

# %%
def train_loop(
    model,
    train_loader,
    val_loader,
    num_epochs,
    early_stopping,
    optimizer: torch.optim.Adam,
    scheduler=None,
    save_path='best_model.pth'
):
    model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
        
    best_val_metric = float('-inf')
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'lr': []
    }

    def plot_metrics(epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, epoch + 1)

        ax1.plot(epochs, history['train_loss'], label='Train Loss')
        ax1.plot(epochs, history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['val_auc'], label='Val ROC AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('Validation ROC AUC Score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_targets.extend(labels.cpu().numpy().flatten())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        val_auc = roc_auc_score(all_targets, all_probs)
        history['val_auc'].append(val_auc)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        print(
            f'Epoch {epoch:3d}/{num_epochs} | '
            f'Train Loss: {epoch_train_loss:.4f} | '
            f'Val Loss: {epoch_val_loss:.4f} | '
            f'Val ROC AUC: {val_auc:.4f} | '
            f'LR: {current_lr:.6f}'
        )

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            torch.save(best_model_state, save_path)
            print(f'New best model saved (val_auc: {best_val_metric:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'Early stopping triggered after {epoch} epochs.')
                break

        if epoch % 10 == 0:
            plot_metrics(epoch)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Best model loaded (val_auc: {best_val_metric:.4f})')
    else:
        print('Warning: No best model state was saved.')

    return model, history

# %%
def make_submission(model, test_loader, df_test, save_path='submission.csv'):
    model.to(DEVICE)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs.flatten())

    submission = pd.DataFrame({
        'id': df_test['id'],
        'y_pred': all_probs
    })

    submission.to_csv(save_path, index=False)
    print(f'Submission saved to {save_path}')

    return submission

# make_submission(model, test_dataloader, "../submissions/subm.csv")

# %% [markdown]
# ### Just CV

# %%
class WildImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform:transforms.Compose=None, has_label=True):
        self.df = df
        self.has_label = has_label
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        label = 0
        if self.has_label:
            label = self.df.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label
    
IN_SHAPE = (256, 256)

train_transforms = transforms.Compose([
    transforms.Resize((int(IN_SHAPE[0] * 1.15), int(IN_SHAPE[1] * 1.15))),  # чуть больше для кропа
    transforms.RandomCrop(IN_SHAPE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),  # таблицы/текст иногда вертикальные
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomGrayscale(p=0.15),  # важно: текст/таблицы часто ч/б
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
    # RandomErasing после ToTensor — имитирует частичное перекрытие
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

test_transforms = transforms.Compose([
    transforms.Resize(IN_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# %%
df_train, df_val = train_test_split(df, test_size=0.3, random_state=42, shuffle=True, stratify=df['label'])

# %%
train_dataset = WildImageDataset(df_train, train_transforms)
val_dataset = WildImageDataset(df_val, test_transforms)
test_dataset = WildImageDataset(df_test, test_transforms, False)

train_dataloader = DataLoader(train_dataset, 512, shuffle=True, num_workers=16, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, 512, shuffle=False, num_workers=16, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, 512, shuffle=False)

# %%
img_idx = np.random.randint(0, len(train_dataset))
img = train_dataset[img_idx][0].numpy()
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)

# %% [markdown]
# #### VladNet

# %%
class VladBinaryCVNet(nn.Module):
    def __init__(
            self,
            img_shape: tuple[int, int],
            in_ch: int,
            hidden_ch: list[int],
            hidden_fc: list[int],
            conv_dropout = 0.2,
            fc_dropout = 0.3
        ):
        super(VladBinaryCVNet, self).__init__()

        self.img_shape = img_shape
        self.in_ch = in_ch

        conv_layers = [
            nn.Conv2d(in_ch, hidden_ch[0], 3, 1, 1),
            nn.BatchNorm2d(hidden_ch[0]),
            nn.ReLU(),
            nn.Conv2d(hidden_ch[0], hidden_ch[0], 3, 1, 1),
            nn.BatchNorm2d(hidden_ch[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(conv_dropout)
        ]
        last_ch = hidden_ch[0]
        for ch in hidden_ch[1:]:
            conv_layers.extend([
                nn.Conv2d(last_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(conv_dropout)
            ])
            last_ch = ch

        self.conv_net = nn.Sequential(*conv_layers)
        
        fc_layers = []
        last_dim = self._shape_after_conv(self.conv_net)
        for dim in hidden_fc:
            fc_layers.extend([
                nn.Linear(last_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout1d(fc_dropout)
            ])
            last_dim = dim
        fc_layers.extend([
            nn.Linear(last_dim, 1)
        ])

        self.fc_net = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = self.conv_net(x)
        flatten = nn.Flatten()
        x = flatten(x)
        x = self.fc_net(x)
        return x


    def _shape_after_conv(self, conv_block: nn.Module):
        A = torch.zeros(size=(1, self.in_ch, *self.img_shape))
        A = conv_block(A)
        flatten = nn.Flatten()
        A = flatten(A)
        return A.shape[1]

VladBinaryCVNet(
    img_shape=IN_SHAPE,
    in_ch=3,
    hidden_ch=[32, 64, 128],
    hidden_fc=[64],
    conv_dropout=0.1
)    

# %%
model = VladBinaryCVNet(
    img_shape=IN_SHAPE,
    in_ch=3,
    hidden_ch=[32, 64, 128, 256],
    hidden_fc=[],
    conv_dropout=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

# model, _ = train_loop(
#     model,
#     train_dataloader,
#     val_dataloader,
#     200,
#     20,
#     optimizer,
#     scheduler,
#     "../models/best_model.pth"
# )


# %% [markdown]
# #### Pretrained models via timm

# %%
# import timm
# import torch
# import torch.nn as nn 

# # Вариант 2: через timm (проще)
# model = timm.create_model(
#     'vit_large_patch14_dinov2.lvd142m',  
#     pretrained=True,
#     num_classes=1,  # бинарная классификация
#     drop_rate=0.3,
# )

# %%
# train_transform = A.Compose([
#     # Геометрические (умеренные — не хотим делать текст нераспознаваемым)
#     A.RandomResizedCrop(
#         height=518,  # native resolution для DINOv2 ViT-L/14
#         width=518,
#         scale=(0.5, 1.0),    # важно: не слишком агрессивный crop
#         ratio=(0.75, 1.33),
#         interpolation=2,      # INTER_CUBIC
#     ),
#     A.HorizontalFlip(p=0.5),
    
#     # НЕ используем вертикальный флип — текст станет нераспознаваемым
#     # и модель потеряет важный сигнал
    
#     A.ShiftScaleRotate(
#         shift_limit=0.05,
#         scale_limit=0.1,
#         rotate_limit=10,      # малый угол — товары обычно ровно сфоткали
#         border_mode=0,
#         p=0.3
#     ),
    
#     # Цветовые (умеренные)
#     A.OneOf([
#         A.ColorJitter(
#             brightness=0.2,
#             contrast=0.2,
#             saturation=0.2,
#             hue=0.05,
#         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#         ),
#     ], p=0.5),
    
#     # Качество изображения (имитация реальных условий маркетплейса)
#     A.OneOf([
#         A.ImageCompression(quality_lower=50, quality_upper=95, p=1.0),
#         A.GaussianBlur(blur_limit=(3, 5), p=1.0),
#         A.GaussNoise(var_limit=(5, 25), p=1.0),
#     ], p=0.3),
    
#     # Важная аугментация для вашей задачи:
#     # имитация watermark/overlay — может превратить нормальное фото 
#     # в "нерелевантное", поэтому используем ОСТОРОЖНО
    
#     # Cutout — помогает модели не цепляться за один регион
#     A.CoarseDropout(
#         max_holes=3,
#         max_height=50,
#         max_width=50,
#         min_holes=1,
#         fill_value=0,
#         p=0.2
#     ),
    
#     # Нормализация
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
#     ToTensorV2()
# ])

# # ===== VALIDATION/TEST =====
# val_transform = A.Compose([
#     A.Resize(height=518, width=518, interpolation=2),
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
#     ToTensorV2()
# ])

# %%
# import torchvision.transforms.v2 as T

# # Mixup + CutMix — очень помогают на бинарной классификации
# from timm.data.mixup import Mixup

# mixup_fn = Mixup(
#     mixup_alpha=0.3,       # не слишком агрессивно для бинарной задачи
#     cutmix_alpha=1.0,
#     cutmix_minmax=None,
#     prob=0.5,              # применяем к 50% батчей
#     switch_prob=0.5,       # 50/50 между mixup и cutmix
#     mode='batch',
#     label_smoothing=0.1,
#     num_classes=2           # для mixup нужно 2 класса
# )

# %%
# class FocalBCELoss(nn.Module):
#     """Focal Loss — критичен при дисбалансе классов"""
#     def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.pos_weight = pos_weight
    
#     def forward(self, logits, targets):
#         bce = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, 
#             pos_weight=self.pos_weight,
#             reduction='none'
#         )
#         probs = torch.sigmoid(logits)
#         p_t = probs * targets + (1 - probs) * (1 - targets)
#         focal_weight = (1 - p_t) ** self.gamma
        
#         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#         loss = alpha_t * focal_weight * bce
        
#         return loss.mean()

# # Если дисбаланс сильный (мало нерелевантных):
# pos_weight = torch.tensor([(df['label'] == 0).sum() / (df['label'] == 1).sum()]).cuda()
# criterion = FocalBCELoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

# # Если дисбаланс умеренный — просто BCE с label smoothing:
# # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# %%
# from torch.cuda.amp import autocast, GradScaler
# from sklearn.metrics import roc_auc_score

# def train_one_epoch(model, loader, optimizer, scheduler, criterion, 
#                      scaler, mixup_fn=None, epoch=0):
#     model.train()
#     all_preds, all_targets = [], []
    
#     for batch_idx, (images, targets) in enumerate(loader):
#         images = images.cuda(non_blocking=True)
#         targets = targets.cuda(non_blocking=True).float()
        
#         # Mixup/CutMix (только для train)
#         if mixup_fn is not None:
#             # Для mixup нужны one-hot targets
#             targets_onehot = torch.zeros(len(targets), 2).cuda()
#             targets_onehot.scatter_(1, targets.long().unsqueeze(1), 1)
#             images, targets_onehot = mixup_fn(images, targets_onehot)
#             use_mixup = True
#         else:
#             use_mixup = False
        
#         with autocast(dtype=torch.bfloat16):  # bf16 на сильном ПК
#             logits = model(images).squeeze(-1)
            
#             if use_mixup:
#                 # Soft targets от mixup
#                 soft_targets = targets_onehot[:, 1]  # probability of class 1
#                 loss = nn.functional.binary_cross_entropy_with_logits(
#                     logits, soft_targets
#                 )
#             else:
#                 loss = criterion(logits, targets)
        
#         scaler.scale(loss).backward()
        
#         # Gradient clipping
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)
#         scheduler.step()
        
#         # Для метрик (без mixup)
#         if not use_mixup:
#             with torch.no_grad():
#                 preds = torch.sigmoid(logits).cpu()
#                 all_preds.extend(preds.numpy())
#                 all_targets.extend(targets.cpu().numpy())
    
#     if all_preds:
#         epoch_auc = roc_auc_score(all_targets, all_preds)
#         return epoch_auc
#     return None

# @torch.no_grad()
# def validate(model, loader):
#     model.eval()
#     all_preds, all_targets = [], []
    
#     for images, targets in loader:
#         images = images.cuda(non_blocking=True)
        
#         with autocast(dtype=torch.bfloat16):
#             logits = model(images).squeeze(-1)
        
#         preds = torch.sigmoid(logits).cpu().numpy()
#         all_preds.extend(preds)
#         all_targets.extend(targets.numpy())
    
#     auc = roc_auc_score(all_targets, all_preds)
#     return auc

# %% [markdown]
# #### ResNet

# %%
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
# model.fc = nn.Linear(model.fc.in_features, 1)

# for param in model.parameters():
#     param.requires_grad = False

# # размораживаем только голову
# for param in model.fc.parameters():
#     param.requires_grad = True
        
# optimizer = torch.optim.Adam([
#     {"params": model.layer4.parameters(), "lr": 1e-4},
#     {"params": model.fc.parameters(), "lr": 1e-3}
# ])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='max',
#     factor=0.5,
#     patience=2
# )

# model, _ = train_loop(
#     model,
#     train_dataloader,
#     val_dataloader,
#     200,
#     20,
#     optimizer,
#     scheduler,
#     "../models/best_model.pth"
# )

# %%
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(DEVICE)
model.fc = nn.Linear(model.fc.in_features, 1)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)
if not os.path.exists(f'../models/best_my_resnet.pth'):
    model, _ = train_loop(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=100,
        early_stopping=10,
        optimizer=optimizer,
        scheduler=None,
        save_path="../models/best_my_resnet.pth"
    )

# %%
class TTATransforms:
    """Набор TTA-трансформаций для инференса."""
    
    def __init__(self, input_size=(256, 256)):
        self.input_size = input_size
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.base = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.tta_transforms = [
            # 0: original
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 1: horizontal flip
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 2: slight zoom (center crop из увеличенного)
            transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.15), int(input_size[1] * 1.15))),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 3: horizontal flip + zoom
            transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.15), int(input_size[1] * 1.15))),
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 4: grayscale (3-channel)
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        ]
    
    def __len__(self):
        return len(self.tta_transforms)
    
    def __getitem__(self, idx):
        return self.tta_transforms[idx]


class TTAImageDataset(Dataset):
    """Датасет, который для каждого изображения применяет одну конкретную TTA-трансформацию."""
    
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose, has_label=True):
        self.df = df
        self.has_label = has_label
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        label = 0
        if self.has_label:
            label = self.df.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label


def predict_with_tta(model, df, tta_transforms: TTATransforms, batch_size=512, 
                      num_workers=16, has_label=False, tta_mode='mean'):
    """
    Прогоняет модель через все TTA-трансформации и усредняет предсказания.
    
    Args:
        model: обученная модель
        df: датафрейм с img_path (и label если has_label=True)
        tta_transforms: объект TTATransforms
        batch_size: размер батча
        num_workers: количество воркеров
        has_label: есть ли метки
        tta_mode: 'mean' - среднее вероятностей, 'gmean' - геом. среднее
    
    Returns:
        all_probs: numpy array усреднённых вероятностей
        all_targets: numpy array меток (если has_label=True, иначе None)
    """
    model.to(DEVICE)
    model.eval()
    
    n_samples = len(df)
    n_tta = len(tta_transforms)
    
    # Матрица: [n_tta, n_samples]
    tta_probs = np.zeros((n_tta, n_samples))
    all_targets = None
    
    for tta_idx in range(n_tta):
        transform = tta_transforms[tta_idx]
        dataset = TTAImageDataset(df, transform, has_label=has_label)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
        
        batch_probs = []
        batch_targets = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                batch_probs.extend(probs)
                if has_label:
                    batch_targets.extend(labels.numpy().flatten())
        
        tta_probs[tta_idx] = np.array(batch_probs)
        
        if has_label and all_targets is None:
            all_targets = np.array(batch_targets)
        
        print(f'  TTA {tta_idx + 1}/{n_tta} done')
    
    # Усреднение
    if tta_mode == 'mean':
        final_probs = tta_probs.mean(axis=0)
    elif tta_mode == 'gmean':
        # Геометрическое среднее — часто лучше для вероятностей
        from scipy.stats import gmean
        final_probs = gmean(tta_probs + 1e-8, axis=0)
    else:
        raise ValueError(f"Unknown tta_mode: {tta_mode}")
    
    return final_probs, all_targets


def validate_with_tta(model, df_val, tta_transforms: TTATransforms, batch_size=512, num_workers=16):
    """Валидация с TTA — показывает ROC AUC."""
    print("Validating with TTA...")
    probs, targets = predict_with_tta(
        model, df_val, tta_transforms, batch_size, num_workers, has_label=True
    )
    auc = roc_auc_score(targets, probs)
    print(f"Validation ROC AUC with TTA: {auc:.4f}")
    return auc, probs, targets


def make_submission_tta(model, df_test, tta_transforms: TTATransforms, 
                        batch_size=512, num_workers=16, save_path='submission.csv'):
    """Создание сабмишена с TTA."""
    print("Making submission with TTA...")
    probs, _ = predict_with_tta(
        model, df_test, tta_transforms, batch_size, num_workers, has_label=False
    )
    
    submission = pd.DataFrame({
        'id': df_test['id'],
        'y_pred': probs
    })
    submission.to_csv(save_path, index=False)
    print(f'Submission saved to {save_path}')
    
    return submission

# %%
# tta = TTATransforms(input_size=IN_SHAPE)

# model = models.resnet34(weights=None).to(DEVICE)
# model.fc = nn.Linear(model.fc.in_features, 1)
# model.load_state_dict(torch.load("../models/best_my_resnet.pth"))
# model = model.to(DEVICE)

# # Сравниваем: без TTA vs с TTA
# print("=== Without TTA ===")
# model.eval()
# all_probs_no_tta = []
# all_targets_no_tta = []
# with torch.no_grad():
#     for inputs, labels in val_dataloader:
#         inputs = inputs.to(DEVICE)
#         outputs = model(inputs)
#         probs = torch.sigmoid(outputs).cpu().numpy().flatten()
#         all_probs_no_tta.extend(probs)
#         all_targets_no_tta.extend(labels.numpy().flatten())

# auc_no_tta = roc_auc_score(all_targets_no_tta, all_probs_no_tta)
# print(f"Val ROC AUC without TTA: {auc_no_tta:.4f}")

# print("\n=== With TTA ===")
# auc_tta, _, _ = validate_with_tta(model, df_val, tta, batch_size=512, num_workers=16)

# print(f"\nImprovement: {auc_tta - auc_no_tta:+.4f}")

# %%
# submission = make_submission_tta(
#     model, df_test, tta,
#     batch_size=512, num_workers=16,
#     save_path='../submissions/subm_tta.csv'
# )

# %%
# model = models.resnet152(weights=None).to(DEVICE)
# model.fc = nn.Linear(model.fc.in_features, 1)
# model.load_state_dict(torch.load("../models/best_my_resnet.pth"))
# make_submission(model, test_dataloader, df_test, "../submissions/resnet_subm.csv")

# %% [markdown]
# #### EfNet

# %%
# import timm

# model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=1)

# # # Сначала посмотрим структуру
# # print("=== Структура модели ===")
# # for name, _ in model.named_parameters():
# #     print(name)

# # Заморозка всего
# for param in model.parameters():
#     param.requires_grad = False

# # Размораживаем последние блоки + голову
# unfreeze_keywords = ['blocks.5', 'blocks.4', 'classifier', 'conv_head', 'bn2']
# for name, param in model.named_parameters():
#     if any(kw in name for kw in unfreeze_keywords):
#         param.requires_grad = True

# # Группируем БЕЗ пересечений — приоритет: classifier/head > blocks.5 > blocks.4
# head_params = []
# blocks5_params = []
# blocks4_params = []

# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         continue
#     # Сначала проверяем head (приоритетнее)
#     if 'classifier' in name:
#         head_params.append(param)
#     # conv_head и bn2 — это часть stem/head, НЕ часть blocks
#     elif 'conv_head' in name or 'bn2' in name:
#         head_params.append(param)
#     elif 'blocks.5' in name:
#         blocks5_params.append(param)
#     elif 'blocks.4' in name:
#         blocks4_params.append(param)

# # Проверяем что ничего не потеряли
# total_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
# total_grouped = len(head_params) + len(blocks5_params) + len(blocks4_params)
# print(f"\nTrainable params: {total_trainable}, Grouped: {total_grouped}")
# assert total_trainable == total_grouped, "Некоторые параметры не попали в группы!"

# optimizer = torch.optim.AdamW([
#     {"params": blocks4_params, "lr": 1e-5},
#     {"params": blocks5_params, "lr": 5e-5},
#     {"params": head_params, "lr": 1e-3},
# ], weight_decay=0.01)

# data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
# print(f"Рекомендуемый input size: {data_cfg['input_size']}")
# print(f"Mean: {data_cfg['mean']}")
# print(f"Std: {data_cfg['std']}")

# %%
model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=1)

# 1) Замораживаем всё
for p in model.parameters():
    p.requires_grad = False

# 2) Размораживаем голову и последние блоки
for p in model.classifier.parameters():
    p.requires_grad = True

for p in model.conv_head.parameters():
    p.requires_grad = True

for p in model.bn2.parameters():
    p.requires_grad = True

# Разморозить последние stage'и backbone
for idx in [4, 5]:
    if idx < len(model.blocks):
        for p in model.blocks[idx].parameters():
            p.requires_grad = True

# 3) Собираем параметры без пересечений
head_params = []
blocks5_params = []
blocks4_params = []

head_params.extend(list(model.classifier.parameters()))
head_params.extend(list(model.conv_head.parameters()))
head_params.extend(list(model.bn2.parameters()))

if len(model.blocks) > 5:
    blocks5_params.extend(list(model.blocks[5].parameters()))

if len(model.blocks) > 4:
    blocks4_params.extend(list(model.blocks[4].parameters()))

# 4) Проверка, что всё trainable попало в группы
trainable_params = [p for p in model.parameters() if p.requires_grad]
grouped_params = head_params + blocks5_params + blocks4_params

assert len(trainable_params) == len(grouped_params), "Некоторые trainable параметры не попали в группы!"

# 5) Optimizer
optimizer = torch.optim.AdamW(
    [
        {"params": blocks4_params, "lr": 1e-5},
        {"params": blocks5_params, "lr": 5e-5},
        {"params": head_params, "lr": 1e-3},
    ],
    weight_decay=0.01,
)

# 6) Data config
data_cfg = timm.data.resolve_model_data_config(model)
print("Рекомендуемый input size:", data_cfg["input_size"])
print("Mean:", data_cfg["mean"])
print("Std:", data_cfg["std"])

# 7) Полезная проверка
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Frozen params:", sum(p.numel() for p in model.parameters() if not p.requires_grad))

# %%
IN_SHAPE = (384, 384)

train_transforms = transforms.Compose([
    transforms.Resize((int(IN_SHAPE[0] * 1.15), int(IN_SHAPE[1] * 1.15))),  # чуть больше для кропа
    transforms.RandomCrop(IN_SHAPE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),  # таблицы/текст иногда вертикальные
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomGrayscale(p=0.15),  # важно: текст/таблицы часто ч/б
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
    # RandomErasing после ToTensor — имитирует частичное перекрытие
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

test_transforms = transforms.Compose([
    transforms.Resize(IN_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# %%
df_train, df_val = train_test_split(df, test_size=0.3, random_state=42, shuffle=True, stratify=df['label'])

# %%
train_dataset = WildImageDataset(df_train, train_transforms)
val_dataset = WildImageDataset(df_val, test_transforms)
test_dataset = WildImageDataset(df_test, test_transforms, False)

train_dataloader = DataLoader(train_dataset, 16, shuffle=True, num_workers=8, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, 16, shuffle=False, num_workers=8, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, 16, shuffle=False)

# %%
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)
if not os.path.exists(f'../models/best_my_efnet.pth'):
    model, _ = train_loop(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=100,
        early_stopping=10,
        optimizer=optimizer,
        scheduler=None,
        save_path="../models/best_my_efnet.pth"
    )

# %%
class TTATransforms:
    """Набор TTA-трансформаций для инференса."""
    
    def __init__(self, input_size=(300, 300)):
        self.input_size = input_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        self.base = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.tta_transforms = [
            # 0: original
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 1: horizontal flip
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 2: slight zoom (center crop из увеличенного)
            transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.15), int(input_size[1] * 1.15))),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 3: horizontal flip + zoom
            transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.15), int(input_size[1] * 1.15))),
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            # 4: grayscale (3-channel)
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        ]
    
    def __len__(self):
        return len(self.tta_transforms)
    
    def __getitem__(self, idx):
        return self.tta_transforms[idx]
   


# %%
tta = TTATransforms(input_size=IN_SHAPE)

model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=1)
model.load_state_dict(torch.load("../models/best_my_efnet.pth"))
model = model.to(DEVICE)

# Сравниваем: без TTA vs с TTA
print("=== Without TTA ===")
model.eval()
all_probs_no_tta = []
all_targets_no_tta = []
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_probs_no_tta.extend(probs)
        all_targets_no_tta.extend(labels.numpy().flatten())

auc_no_tta = roc_auc_score(all_targets_no_tta, all_probs_no_tta)
print(f"Val ROC AUC without TTA: {auc_no_tta:.4f}")

print("\n=== With TTA ===")
auc_tta, _, _ = validate_with_tta(model, df_val, tta, batch_size=512, num_workers=16)

print(f"\nImprovement: {auc_tta - auc_no_tta:+.4f}")

# %%
# make_submission(model, test_dataloader, df_test, "../submissions/efnet_subm.csv")

# %%
submission = make_submission_tta(
    model, df_test, tta,
    batch_size=512, num_workers=16,
    save_path='../submissions/subm_tta.csv'
)

# %% [markdown]
# #### Resnet with tpriplet loss

# %%
class ResNetMetric(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Linear(in_features, emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)                 # (B, 2048)
        emb = self.embedding(feat)              # (B, emb_dim)
        emb = F.normalize(emb, p=2, dim=1)      # важно для metric learning
        logits = self.classifier(emb)          # (B, 1)
        return logits, emb

# %%
def build_triplets(embeddings, labels):
    """
    embeddings: (B, D)
    labels: (B,) или (B,1)
    """
    labels = labels.view(-1)
    triplets = []

    for i in range(len(labels)):
        anchor_label = labels[i].item()

        pos_idx = (labels == anchor_label).nonzero(as_tuple=True)[0]
        neg_idx = (labels != anchor_label).nonzero(as_tuple=True)[0]

        # нужно хотя бы 2 объекта этого класса для positive
        if len(pos_idx) < 2 or len(neg_idx) < 1:
            continue

        pos_choices = pos_idx[pos_idx != i]
        if len(pos_choices) == 0:
            continue

        p = pos_choices[random.randint(0, len(pos_choices) - 1)]
        n = neg_idx[random.randint(0, len(neg_idx) - 1)]
        triplets.append((i, p.item(), n.item()))

    return triplets

# %%
def train_loop_metric(
    model,
    train_loader,
    val_loader,
    num_epochs,
    early_stopping,
    optimizer,
    scheduler=None,
    save_path='best_model.pth',
    alpha=0.2
):
    model.to(DEVICE)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_triplet = nn.TripletMarginLoss(margin=0.3)

    best_val_metric = float('-inf')
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'lr': []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()

            logits, emb = model(inputs)
            loss_cls = criterion_cls(logits, labels)

            # triplet loss
            triplets = build_triplets(emb, labels)
            if len(triplets) > 0:
                a_idx = torch.tensor([t[0] for t in triplets], device=DEVICE)
                p_idx = torch.tensor([t[1] for t in triplets], device=DEVICE)
                n_idx = torch.tensor([t[2] for t in triplets], device=DEVICE)

                anchor = emb[a_idx]
                positive = emb[p_idx]
                negative = emb[n_idx]

                loss_triplet = criterion_triplet(anchor, positive, negative)
            else:
                loss_triplet = torch.tensor(0.0, device=DEVICE)

            loss = loss_cls + alpha * loss_triplet
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                logits, emb = model(inputs)
                loss = criterion_cls(logits, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_targets.extend(labels.cpu().numpy().flatten())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        val_auc = roc_auc_score(all_targets, all_probs)
        history['val_auc'].append(val_auc)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        print(
            f'Epoch {epoch:3d}/{num_epochs} | '
            f'Train Loss: {epoch_train_loss:.4f} | '
            f'Val Loss: {epoch_val_loss:.4f} | '
            f'Val ROC AUC: {val_auc:.4f} | '
            f'LR: {current_lr:.6f}'
        )

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(best_model_state, save_path)
            print(f'New best model saved (val_auc: {best_val_metric:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'Early stopping triggered after {epoch} epochs.')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Best model loaded (val_auc: {best_val_metric:.4f})')

    return model, history

# %%
model = ResNetMetric(emb_dim=256).to(DEVICE)
for name, param in model.named_parameters():
    if "layer4" in name or "embedding" in name or "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        
optimizer = torch.optim.Adam([
    {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
    {"params": model.embedding.parameters(), "lr": 1e-3},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)
if not os.path.exists(f'../models/best_my_resnet_metric.pth'):
    model, _ = train_loop_metric(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=100,
        early_stopping=10,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path="../models/best_my_resnet_metric.pth",
        alpha=0.2
    )

# %%
def make_submission_metric(
    model,
    test_loader,
    df_test,
    save_path='submission.csv'
):
    model.to(DEVICE)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # универсально: (img, label) или (img,)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(DEVICE)

            logits, _ = model(inputs)  # <-- важно
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.flatten())

    submission = pd.DataFrame({
        'id': df_test['id'].values,
        'y_pred': all_probs
    })

    submission.to_csv(save_path, index=False)
    print(f'Submission saved to {save_path}')

    return submission

# %%
# model = ResNetMetric(emb_dim=256)
# model.load_state_dict(torch.load("../models/best_my_resnet_metric.pth"))
# model = model.to(DEVICE)
# model.eval()

# make_submission_metric(
#     model,
#     test_dataloader,
#     df_test,
#     save_path="../submissions/m_subm.csv"
# )

# %% [markdown]
# ### CLIP tunning

# %%
df.head()

# %%
# model = SentenceTransformer(
#     "sentence-transformers/clip-ViT-B-32-multilingual-v1",
#     trust_remote_code=True,
#     truncate_dim=512,
# )

# train_dict = {
#     "image": df_train["img_abs_path"].tolist(),
#     "text": df_train["text"].tolist(),
# }

# negative_cols = [c for c in df_train.columns if c.startswith("negative_")]
# for c in negative_cols:
#     train_dict[c] = df_train[c].tolist()

# train_dataset = Dst.from_dict(train_dict)

# class PairSimilarityEvaluator(BaseEvaluator):
#     def __init__(self, images, texts, labels, name="val", batch_size=64):
#         self.images = list(images)
#         self.texts = list(texts)
#         self.labels = np.asarray(labels, dtype=np.int32)
#         self.name = name
#         self.batch_size = batch_size

#         self.primary_metric = f"{name}_f1"
#         self.greater_is_better = True

#     def __call__(self, model, output_path=None, epoch=-1, steps=-1):
#         model.eval()

#         with torch.no_grad():
#             img_emb = model.encode(
#                 self.images,
#                 batch_size=self.batch_size,
#                 convert_to_tensor=True,
#                 normalize_embeddings=True,
#                 show_progress_bar=False,
#             )
#             txt_emb = model.encode(
#                 self.texts,
#                 batch_size=self.batch_size,
#                 convert_to_tensor=True,
#                 normalize_embeddings=True,
#                 show_progress_bar=False,
#             )

#             scores = (img_emb * txt_emb).sum(dim=1).detach().cpu().numpy()

#         # threshold sweep for best F1
#         thresholds = np.unique(scores)
#         best_f1 = -1.0
#         best_acc = -1.0
#         best_thr = float(thresholds[0]) if len(thresholds) else 0.0

#         for thr in thresholds:
#             preds = (scores >= thr).astype(np.int32)

#             tp = np.sum((preds == 1) & (self.labels == 1))
#             fp = np.sum((preds == 1) & (self.labels == 0))
#             fn = np.sum((preds == 0) & (self.labels == 1))

#             precision = tp / (tp + fp + 1e-12)
#             recall = tp / (tp + fn + 1e-12)
#             f1 = 2 * precision * recall / (precision + recall + 1e-12)
#             acc = (preds == self.labels).mean()

#             if f1 > best_f1:
#                 best_f1 = float(f1)
#                 best_acc = float(acc)
#                 best_thr = float(thr)

#         metrics = {
#             f"{self.name}_accuracy": best_acc,
#             f"{self.name}_f1": best_f1,
#             f"{self.name}_threshold": best_thr,
#         }
#         return metrics
    

# evaluator = PairSimilarityEvaluator(
#     images=df_val["img_abs_path"],
#     texts=df_val["text"],
#     labels=df_val["label"],
#     name="val",
#     batch_size=64,
# )

# loss = losses.MultipleNegativesRankingLoss(
#     model,
#     hardness_mode="all_negatives",
#     hardness_strength=5.0,
# )

# # ---------- 5) Training args ----------
# training_args = SentenceTransformerTrainingArguments(
#     output_dir="clip-ViT-finetuned",
    
#     num_train_epochs=5,
#     per_device_train_batch_size=64,
#     gradient_accumulation_steps=2,
#     learning_rate=2e-5,
#     warmup_ratio=0.1,

#     fp16=True,
#     bf16=False,

#     # Change 'evaluation_strategy' to 'eval_strategy'
#     eval_strategy="epoch", 
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="val_f1",
#     greater_is_better=True,

#     save_total_limit=2,
#     logging_steps=50,
#     report_to="none",
# )

# # ---------- 6) Trainer ----------
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     evaluator=evaluator,
#     loss=loss,
# )

# trainer.train()

# model.save_pretrained("clip-ViT-finetuned-final")
# print("✅ Fine-tuning завершён!")

# %% [markdown]
# ### Another model with CLIP embedings

# %% [markdown]
# #### Embedings extraction

# %% [markdown]
# ##### CLIP

# %%
if not os.path.exists(f'{DATA_PATH}/train_text_embeddings.npy'):
    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        trust_remote_code=True,
        device=DEVICE,
        truncate_dim=512
    )

    text_batch = df['text'].sample(3).to_list()
    print(model.encode(text_batch, batch_size=len(text_batch)))

    texts = df['text'].to_list()
    img_paths = df['img_abs_path'].to_list()
    batch_size = 256

    text_embeds = []
    image_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        text_batch = texts[i : i + batch_size]
        img_batch = img_paths[i : i + batch_size]

        text_emb = model.encode(
            text_batch,
            normalize_embeddings=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            device=DEVICE
        )

        image_emb = model.encode(
            img_batch,
            normalize_embeddings=True,
            batch_size=len(img_batch),
            show_progress_bar=False,
            device=DEVICE
        )
        
        text_embeds.extend(text_emb)
        image_embeds.extend(image_emb)
        
    np.save(f'{DATA_PATH}/train_text_embeddings.npy', np.array(text_embeds))
    np.save(f'{DATA_PATH}/train_img_embedings.npy', np.array(image_embeds))


    texts = df_test['text'].to_list()
    img_paths = df_test['img_abs_path'].to_list()
    batch_size = 256

    text_embeds = []
    image_embeds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        text_batch = texts[i : i + batch_size]
        img_batch = img_paths[i : i + batch_size]

        text_emb = model.encode(
            text_batch,
            normalize_embeddings=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            device=DEVICE
        )

        image_emb = model.encode(
            img_batch,
            normalize_embeddings=True,
            batch_size=len(img_batch),
            show_progress_bar=False,
            device=DEVICE
        )
        
        text_embeds.extend(text_emb)
        image_embeds.extend(image_emb)
        
    np.save(f'{DATA_PATH}/test_text_embeddings.npy', np.array(text_embeds))
    np.save(f'{DATA_PATH}/test_img_embedings.npy', np.array(image_embeds))

# %% [markdown]
# ##### My model

# %%
# if not os.path.exists(f'{DATA_PATH}/resnet_train_img_embs.npy'):
#     model = models.resnet50(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 1)
#     model.load_state_dict(torch.load('../models/best_my_resnet.pth'))
#     model.fc = nn.Identity()
#     model = model.to(DEVICE)
#     model.eval()

#     df_dataset = WildImageDataset(df, test_transforms)
#     test_df_dataset = WildImageDataset(df_test, test_transforms, False)

#     df_loader = DataLoader(df_dataset, batch_size=256, shuffle=False, num_workers=0)
#     test_df_dataset = DataLoader(test_df_dataset, batch_size=256, shuffle=False, num_workers=0)

#     def extract_embeddings(loader, model, device):
#         all_embeddings = []

#         with torch.no_grad():
#             for batch in tqdm(loader):
#                 # если dataset возвращает (img, label)
#                 if isinstance(batch, (list, tuple)):
#                     images = batch[0]
#                 else:
#                     images = batch

#                 images = images.to(device)

#                 emb = model(images)              # (B, 2048)
#                 emb = emb.flatten(1)

#                 # нормализация (очень желательно)
#                 emb = torch.nn.functional.normalize(emb, p=2, dim=1)

#                 all_embeddings.append(emb.cpu().numpy())

#         return np.concatenate(all_embeddings, axis=0)

#     train_embeddings = extract_embeddings(df_loader, model, DEVICE)
#     test_embeddings = extract_embeddings(test_df_dataset, model, DEVICE)

#     np.save(f'{DATA_PATH}/resnet_train_img_embs.npy', train_embeddings)
#     np.save(f'{DATA_PATH}/resnet_test_img_embs.npy', test_embeddings)

# %%
# if not os.path.exists(f'{DATA_PATH}/resnet_train_img_embs.npy'):
#     def extract_embeddings(loader, model):
#         model.eval()
#         embs = []

#         with torch.no_grad():
#             for batch in tqdm(loader):
#                 images = batch[0].to(DEVICE)
#                 _, emb = model(images)
#                 embs.append(emb.cpu().numpy())

#         return np.concatenate(embs, axis=0)

#     model = ResNetMetric(emb_dim=256)
#     model.load_state_dict(torch.load("../models/best_my_resnet_metric.pth"))
#     model = model.to(DEVICE)
#     model.eval()
    
#     df_dataset = WildImageDataset(df, test_transforms)
#     test_df_dataset = WildImageDataset(df_test, test_transforms, False)

#     df_loader = DataLoader(df_dataset, batch_size=256, shuffle=False, num_workers=0)
#     test_df_dataset = DataLoader(test_df_dataset, batch_size=256, shuffle=False, num_workers=0)

#     train_embeddings = extract_embeddings(df_loader, model)
#     test_embeddings = extract_embeddings(test_df_dataset, model)

#     np.save(f'{DATA_PATH}/resnet_train_img_embs.npy', train_embeddings)
#     np.save(f'{DATA_PATH}/resnet_test_img_embs.npy', test_embeddings)

# %%
train_text_emb = np.load(f'{DATA_PATH}/train_text_embeddings.npy', allow_pickle=True)
train_img_emb = np.load(f'{DATA_PATH}/train_img_embedings.npy', allow_pickle=True)
# train_rn_img_emb = np.load(f'{DATA_PATH}/resnet_train_img_embs.npy', allow_pickle=True)

test_text_emb = np.load(f'{DATA_PATH}/test_text_embeddings.npy', allow_pickle=True)
test_img_emb = np.load(f'{DATA_PATH}/test_img_embedings.npy', allow_pickle=True)
# test_rn_img_emb = np.load(f'{DATA_PATH}/resnet_test_img_embs.npy', allow_pickle=True)

batch_size = 256
train_similarities = []
test_similarities = []

train_l2 = []
test_l2 = []


for i in tqdm(range(0, len(train_text_emb), batch_size), desc="Train Batches"):    
    text_embeds = np.array(train_text_emb[i : i + batch_size])
    image_embeds = np.array(train_img_emb[i : i + batch_size])

    text_embeds = torch.tensor(text_embeds).to(DEVICE)
    image_embeds = torch.tensor(image_embeds).to(DEVICE)

    cos_sims = util.cos_sim(text_embeds, image_embeds)
    diag_sims = torch.diag(cos_sims).cpu().numpy()
    train_similarities.extend(diag_sims)

    l2_dist = torch.norm(text_embeds - image_embeds, dim=1).cpu().numpy()
    train_l2.extend(l2_dist)


for i in tqdm(range(0, len(test_text_emb), batch_size), desc="Test Batches"):    
    text_embeds = np.array(test_text_emb[i : i + batch_size])
    image_embeds = np.array(test_img_emb[i : i + batch_size])

    text_embeds = torch.tensor(text_embeds).to(DEVICE)
    image_embeds = torch.tensor(image_embeds).to(DEVICE)

    cos_sims = util.cos_sim(text_embeds, image_embeds)
    diag_sims = torch.diag(cos_sims).cpu().numpy()
    test_similarities.extend(diag_sims)

    l2_dist = torch.norm(text_embeds - image_embeds, dim=1).cpu().numpy()
    test_l2.extend(l2_dist)


df['text_emb'] = list(train_text_emb)
df['img_emb'] = list(train_img_emb)
# df['rn_img_emb'] = list(train_rn_img_emb)
df['emb_sim'] = train_similarities
df['emb_l2'] = train_l2

df_test['text_emb'] = list(test_text_emb)
df_test['img_emb'] = list(test_img_emb)
# df_test['rn_img_emb'] = list(test_rn_img_emb)
df_test['emb_sim'] = test_similarities
df_test['emb_l2'] = test_l2

# %% [markdown]
# #### FC Network

# %%
class WildDataset(Dataset):
    def __init__(self, df: pd.DataFrame, has_labels=True):
        self.df = df
        self.has_labels = has_labels
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text_emb = np.array(self.df.iloc[idx]['text_emb'], dtype=np.float32)
        img_emb = np.array(self.df.iloc[idx]['img_emb'], dtype=np.float32)
        # rn_img_emb = np.array(self.df.iloc[idx]['rn_img_emb'], dtype=np.float32)
        emb_sim = np.array([self.df.iloc[idx]['emb_sim']], dtype=np.float32)
        emb_l2 = np.array([self.df.iloc[idx]['emb_l2']], dtype=np.float32)
        # ti_emb = text_emb * img_emb
        
        combined = np.concatenate([text_emb, img_emb, emb_sim, emb_l2])
        combined = torch.from_numpy(combined)
        
        if self.has_labels:
            label = self.df.iloc[idx]['label']
            return combined, label
        else:
            return combined, 0

# %%
df_train, df_val = train_test_split(df, test_size=0.3, random_state=42, shuffle=True, stratify=df['label'])

# %%
train_dataset = WildDataset(df_train)
val_dataset = WildDataset(df_val)
test_dataset = WildDataset(df_test, False)

train_dataloader = DataLoader(train_dataset, 1024, shuffle=True, num_workers=16, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, 1024, shuffle=False, num_workers=16, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, 1024, shuffle=False)

input_dim = train_dataset[0][0].shape[0]
input_dim

# %%
class VladBinaryNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3
    ):
        super(VladBinaryNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim=dim
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
        
VladBinaryNet(input_dim)

# %%
model = VladBinaryNet(
    input_dim=input_dim,
    hidden_dims=[256, 128, 64],
    dropout=0.3
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

model, _ = train_loop(
    model,
    train_dataloader,
    val_dataloader,
    200,
    20,
    optimizer,
    scheduler,
    "../models/best_model.pth"
)


# %%
make_submission(model, test_dataloader, df_test, "../submissions/subm.csv")

# %%




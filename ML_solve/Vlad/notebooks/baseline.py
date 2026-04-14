# %% [markdown]
# # Baseline Solution
# 
# ## Описание
# 
# В этом Jupyter-ноутбуке представлено **базовое решение** задачи бинарной классификации: **Определение релевантности изображения для карточки товара.**
# 
# Данное решение:
# - Простое и быстрое в реализации, так как не требует обучения модели с нуля.
# - Использует предобученную модель [**jinaai/jina-clip-v2**](https://huggingface.co/jinaai/jina-clip-v2).
# - Подходит для быстрого старта, улучшения, и поможет разобраться как правильно формировать файл решения.

# %% [markdown]
# ## Imports and constants

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

# %% [markdown]
# Как уже было указано, в данном решении используется предобученная модель **CLIP (Contrastive Language–Image Pretraining)** от OpenAI —  а именно, её усовершенствованная версия от Jina AI:
# [**jinaai/jina-clip-v2**](https://huggingface.co/jinaai/jina-clip-v2).
# 
# Решение основано на анализе семантической схожести пары **изображение — текст** (в нашем случае текст — это название товара и его описание).  
# 
# Модель Jina CLIP v2 расширяет оригинальную архитектуру CLIP, предлагая улучшенное понимание как на английском, так и на множестве других языков, включая русский, а также демонстрирует повышенную точность в задачах сопоставления изображений и текстов. Мы используем модель следующим образом:
# 1. Для каждой пары *изображение — название товара + его описание* вычисляется их **схожесть (similarity score)**. В качестве схожести используем **косинусную близость**.
# 2. Этот скор интерпретируется как **вероятность релевантности** изображения для данной карточки товара.

# %%
MODEL_NAME = "jinaai/jina-clip-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Замените пути на свои при необходимости
DATA_FOLDER = Path(r"../data")
IMAGE_FOLDER = DATA_FOLDER / "images"

print(f"Currently using {DEVICE}")

# %% [markdown]
# ## Model Initialization
# 
# Загружаем модель с huggingface. Будем использовать библиотеку [sentence_transformers](https://sbert.net) 

# %%
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
model.to(DEVICE)

print(f"Model is currently on {model.device}")

# %% [markdown]
# ## Data Loading
# 
# Поскольку в данном решении мы не проводим обучение модели, необходимость в полной обучающей выборке отсутствует.  
# 
# Тем не менее, для оценки качества нашего подхода мы выделим небольшую **валидационную подвыборку** из обучающих данных. С её помошью:
# - Проверим корректность реализации логики инференса.
# - Оценим целевую метрику качества на размеченных данных.
# - Получить приблизительное представление о том, насколько хорошо модель справляется с задачей определения релевантности.
# 
# В дальнейшем эта подвыборка будет обозначаться как `val_df`

# %%
train_df = pd.read_csv(DATA_FOLDER / "train.csv")
test_df = pd.read_csv(DATA_FOLDER / "test.csv")

val_df = train_df.head(500)

# %% [markdown]
# В качестве меры схожности, как было сказано ранее, применяется **косинусная близость (cosine similarity)** между векторными представлениями (эмбеддингами) изображения и текста.
# 
# Следующий код реализует: 
# - Базовую предобработку текстов (конкатенацию названия и описания)
# - Вычисление поэлементного косинусного сходства для соответствующих пар
# 
# Чем ближе значение косинусной близости к 1, тем выше предполагаемая релевантность изображения для данного текстового описания.

# %%
def combine_texts(row):
    return f"Название товара: {row['name']}. Описание товара: {row['description']}"


batch_size = 16
similarities = []

image_paths = [os.path.join(IMAGE_FOLDER, f"{row_id}.jpg") for row_id in val_df["id"]]
texts = val_df.apply(combine_texts, axis=1).tolist()

for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
    text_batch = texts[i : i + batch_size]
    img_batch = image_paths[i : i + batch_size]

    text_embeds = model.encode(
        text_batch,
        normalize_embeddings=True,
        batch_size=len(text_batch),
        show_progress_bar=False,
    )
    image_embeds = model.encode(
        img_batch,
        normalize_embeddings=True,
        batch_size=len(img_batch),
        show_progress_bar=False,
    )

    text_embeds = torch.tensor(text_embeds).to(DEVICE)
    image_embeds = torch.tensor(image_embeds).to(DEVICE)

    cos_sims = util.cos_sim(text_embeds, image_embeds)
    diag_sims = torch.diag(cos_sims).cpu().numpy()

    similarities.extend(diag_sims)

val_df = val_df.copy()
val_df["similarity_score"] = similarities

# %% [markdown]
# Теперь мы можем оценить качество полученных предсказаний с помощью метрики **ROC AUC**.
# 
# ROC AUC интерпретируется как вероятность того, что случайно выбранная положительная пара (релевантное изображение) получит от модели более высокий скор, чем случайно выбранная отрицательная пара (нерелевантное изображение).  
# Значение:
# - **ROC AUC = 0.5** — модель не лучше случайного угадывания.
# - **ROC AUC > 0.5** — модель устойчиво различает классы.
# - **ROC AUC < 0.5** — модель систематически "путает" классы: присваивает более высокие скоры нерелевантным изображениям, чем релевантным.

# %%
roc_auc_val = roc_auc_score(val_df["label"], val_df["similarity_score"])
print(f"ROC AUC на валидации:{roc_auc_val: .3f}")

# %% [markdown]
# Аналогично получим схожесть для тестовой выборки.

# %%
image_paths = [os.path.join(IMAGE_FOLDER, f"{row_id}.jpg") for row_id in test_df["id"]]
texts = test_df.apply(combine_texts, axis=1).tolist()

similarities = []

for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
    text_batch = texts[i : i + batch_size]
    img_batch = image_paths[i : i + batch_size]

    text_embeds = model.encode(
        text_batch,
        normalize_embeddings=True,
        batch_size=len(text_batch),
        show_progress_bar=False,
    )
    image_embeds = model.encode(
        img_batch,
        normalize_embeddings=True,
        batch_size=len(img_batch),
        show_progress_bar=False,
    )

    text_embeds = torch.tensor(text_embeds).to(DEVICE)
    image_embeds = torch.tensor(image_embeds).to(DEVICE)

    cos_sims = util.cos_sim(text_embeds, image_embeds)
    diag_sims = torch.diag(cos_sims).cpu().numpy()

    similarities.extend(diag_sims)

test_df["y_pred"] = similarities

# %% [markdown]
# ## Нормализация сходства: приведение скоров к диапазону [0, 1]
# 
# Значения косинусной близости, возвращаемые моделью CLIP, лежат в диапазоне **от -1 до 1**, однако, для совместимости с **требованиями платформы соревнования**, предсказания **должны быть представлены в виде вероятностей**, то есть в диапазоне **[0, 1]**.
# 
# Это преобразование можно выполнить с помощью **сигмоидной функции**:
# 
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}
# $$
# Применение сигмоиды не повлияет на метрику так как **сохраняет порядок** предсказаний — если одно изображение было оценено как более релевантное до преобразования, оно останется более релевантным после.

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


test_df["y_pred"] = sigmoid(test_df["y_pred"])

# %% [markdown]
# Сформируем csv файл решения. Полученный файл можно отправить на платформу.

# %%
test_df[["id", "y_pred"]].to_csv(DATA_FOLDER / "sample_submission.csv", index=None)

# %% [markdown]
# Если весь процесс прошёл удачно, то вы можете ожидать метрику на лидерборде $\approx 0.64$ ROC AUC



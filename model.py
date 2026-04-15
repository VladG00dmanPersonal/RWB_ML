import time

from PIL import Image
import numpy as np

class Model:
    def __init__(self):
        # load model here
        time.sleep(3)
        pass

    def predict(self, name : str, description : str, images : list[Image.Image]) -> np.ndarray:
        return np.array([np.random.random() for _ in range(len(images))])
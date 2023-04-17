from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
import os
from PIL import Image


class FeatureExtractionDataset(Dataset):
    def __init__(self, img_root, data, transform=Compose([
            Resize((224, 224)),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])):
        self.img_root = img_root
        self.transform = transform
        self.data = data

    def __getitem__(self, item):
        image = Image.open(f'{self.img_root}/{self.data.iloc[item, 1]}')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image

    def __len__(self):
        return len(self.data)

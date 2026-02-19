from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, correct_classes, csv_filename, transform):
        print("Reading Image Dataset...")
        self.classes = correct_classes
        self.label_to_idx = {val: idx for idx, val in enumerate(self.classes)}

        print("Reading dataset file paths...")
        self.img_labels = pd.read_csv(f'{csv_filename}', delimiter=',', header=None)
        self.transform = transform
        print("Image Dataset instance created!")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        label = self.img_labels.iloc[index, 0]
        img_name = self.img_labels.iloc[index, 1]
        img_path = f'{img_name}'
        image = Image.open(img_path).convert('RGB')

        image_tensor = self.transform(image)
        image.close()
        return image_tensor, self.label_to_idx[label]
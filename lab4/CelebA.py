import os, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class CelebA(Dataset):
    def __init__(self, img_dir, attr, sz):
        self.img_dir = img_dir
        self.sz = sz
        self.attr = pd.read_csv(attr, sep=r",")
        self.files = []
        cnt = 0
        for ind in self.attr.index.tolist():
          if os.path.exists(os.path.join(self.img_dir, self.attr.loc[ind]["image_id"])):
            self.files.append(ind)
          else:
            cnt += 1
            print(f"{self.attr.loc[ind]["image_id"]} doesn't exist")
        print(f"{cnt} files are absent")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self,i):
        f = self.files[i]
        img = Image.open(os.path.join(self.img_dir, self.attr.loc[f]["image_id"])).resize((self.sz, self.sz))
        img = self.transform(img)
        gender = self.attr.loc[f]["Male"]
        return img, torch.tensor(gender).float()
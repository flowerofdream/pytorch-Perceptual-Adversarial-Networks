import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

# facades Dataset
class FacadesDataset(Dataset):
    def __init__(self, root, image_size=256, mode='train', start=0, stop=400, augmentation_prob=0.4):
        self.root = root
        self.image_size = image_size
        self.files_names = os.listdir(root)
        self.imag = []
        for name in self.files_names:
            X_img, Y_img =self.pix2pix(name)
            self.imag.append((X_img, Y_img))

    def __getitem__(self, index):
        X_img, Y_img = self.imag[index]
        return X_img, Y_img

    def __len__(self):
        return len(self.imag)

    def pix2pix(self, name):
        C, H, W = 3, self.image_size, self.image_size
        img = Image.open(self.root + name)
        img = img.resize((2 * W, H), Image.ANTIALIAS)
        img = np.array(img)
        y = img[:, 0:W, :]
        X = img[:, W:, :]
        return X, y


if __name__ == "__main__":
    test = FacadesDataset("../datasets/facades/train/")
    loader = DataLoader(dataset=test, batch_size=4, shuffle=True)
    for img, i in loader:
        print(img.shape)
        break

import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from torchvision import transforms
from torch.utils.data import Dataset

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CelebA(Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, gender=None, target=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = np.concatenate(dataframe.labels.values).astype(float)
        gender_id = 20

        if gender is not None:
            if target is not None:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                target_idx = np.where(label_np[:, target_id] == target)[0]
                idx = list(set(gender_idx) & set(target_idx))
                self.file_names = self.file_names[idx]
                self.labels = np.concatenate(dataframe.labels.values[idx]).astype(float)
            else:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                self.file_names = self.file_names[gender_idx]
                self.labels = np.concatenate(dataframe.labels.values[gender_idx]).astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label[self.target_id]


def get_loader(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA(df, data_path, target_id, transform=tfms, gender=gender, target=target)

    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=3, drop_last=True)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3)

    return dloader

def evaluate(model, model_linear, dataloader):
    y_scores = []
    y_true = []
    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.cuda(), target.float().cuda()

        feat = model(inputs)
        pred = model_linear(feat).detach()

        y_scores.append(pred[:, 0].data.cpu().numpy())
        y_true.append(target.data.cpu().numpy())

    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    ap = average_precision_score(y_true, y_scores)
    return ap, np.mean(y_scores)

def BCELoss(pred, target):
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

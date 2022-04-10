import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(data, test_size=None,
                         valid_size=None,
                         random_state=42,
                         stratify=None):
    train_valid, test = train_test_split(data, test_size=test_size, random_state=random_state,  shuffle=True,
                                         stratify=data[stratify] if stratify else None)
    train, valid = train_test_split(train_valid, test_size=valid_size/(1-test_size),
                                    random_state=random_state, shuffle=True,
                                    stratify=train_valid[stratify] if stratify else None)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, valid, test


class WoodDataset(Dataset):
    """Велосипед вместо ImageFolder"""

    def __init__(self,
                 img_dir: str = None,
                 is_test: bool = False,
                 task_type: str = 'classification',
                 dataset_role: str = 'train',
                 train_val_test_split_params: dict = None,
                 img_transforms: transforms.Compose = None):
        """
        :param img_dir: путь до папки с картинками. Считается, что структура такая же, как и в гугл диске.
        :param is_test: является ли этот датасет оценочным (его пушить на каггл)?
        """
        if task_type == 'classification':
            target_mapping = {'drova': [0, 0, 1], '1': [1, 0, 0], '3': [0, 1, 0]}
        elif task_type == 'ranking':
            target_mapping = {'drova': [0, 0], '1': [1, 1], '3': [1, 0]}
        else:
            raise ValueError('Выбери "classification" или "ranking"')
        if img_transforms is not None:
            self.img_transforms = img_transforms
        else:
            # Стоит заменять на параметры своего датасета
            self.img_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.data = pd.DataFrame(columns=['img_path', 'target'])
        if is_test:
            self.data.loc[:, 'img_path'] = os.listdir(os.path.join(img_dir, 'test'))
        else:
            all_data = []
            for trg in ['drova', '1', '3']:
                files = os.listdir(os.path.join(img_dir, 'train', trg))
                all_data.append(pd.DataFrame({'img_path': files, 'target': [trg] * len(files)}))
            self.data = pd.concat(all_data, ignore_index=True)
            train, val, test = train_val_test_split(self.data, **train_val_test_split_params)
            if dataset_role == 'train':
                self.data = train
            elif dataset_role == 'valid':
                self.data = val
            elif dataset_role == 'test':
                self.data = test
            else:
                raise ValueError('Выбери "train", "valid" или "test"')
            self.data.loc[:, 'target'] = self.data.target.replace(target_mapping)
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.loc[idx, 'img_path']
        in_img = Image.open(path)
        in_img = self.img_transforms(in_img)
        target = torch.tensor(self.data.loc[idx, 'target'])
        return {'image': in_img, 'target': target, 'img_path': path}

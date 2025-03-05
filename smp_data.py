# To load training and testing data
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import datasets, transforms, utils
from scipy.stats import zscore

class bili_data(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

        self.use_img = True  # False
        self.text = True  # False#
        self.num = True  # False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = str(self.data.iloc[idx, 0])

        if self.use_img:
            img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]) + '.jpg')
            try:
                image = Image.open(img_name).convert("RGB")
                image = self.transform(image)
            except:
                image = torch.zeros(3, 224, 224)
                print('no img{}'.format(id))
        else:
            image = torch.zeros(3, 224, 224)

        if self.text:
            category = str(self.data.iloc[idx, 1])
            title = str(self.data.iloc[idx, 2])

        else:
            category = '0'
            title = '0'
            print('no text{}'.format(id))

        text = [category, title]

        if self.num:
            num = torch.from_numpy(self.data.iloc[idx, 3:11].values.astype(np.float32))
        else:
            num = torch.zeros(8)
            print('no num{}'.format(id))
        meta = num

        label = torch.from_numpy(np.array(self.data.iloc[idx, -1])).float().view(1)

        sample = {'img': image, 'meta': meta, 'id': id, 'label': label, 'text': text, 'category': category,
                  'title': title}

        return sample


class bili_data_lstm(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file, header=None)
        self.data.iloc[:, 9] = zscore(self.data.iloc[:, 9])

        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        self.use_img = True  # False
        self.text = True  # False#
        self.num = True  # False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = str(int(self.data.iloc[idx, 0]))
        if self.use_img:
            img_name = os.path.join(self.root_dir, str(int(self.data.iloc[idx, 0])) + '.jpg')
            try:
                image = Image.open(img_name).convert("RGB")
                image = self.transform(image)
            except:
                image = torch.zeros(3, 224, 224)
                # print('no img{}'.format(id))
        else:
            image = torch.zeros(3, 224, 224)

        if self.text:
            category = str(self.data.iloc[idx, 1])
            # category = '0'
            title = str(self.data.iloc[idx, 2])
            # title = '0'

        else:
            category = '0'
            title = '0'

        text = [category, title]

        if self.num:
            num = torch.from_numpy(self.data.iloc[idx, 3:10].values.astype(np.float32))
            num[1:-1] = 0
        else:
            num = torch.zeros(7)
        meta = num

        # label = torch.from_numpy(self.data.iloc[idx, 10: 40].values.astype(np.float32))
        label_ = self.data.iloc[idx, 10: 40].values.astype(np.float32)
        label = torch.from_numpy(label_)

        # first = self.data.iloc[idx, 9].astype(np.float32)
        # label_coef = torch.from_numpy(label_ / first)

        # sample = {'img': image,
        #           'text': text,
        #           # 'profile_img': profile_img,
        #           'meta': meta,
        #           'id': id,
        #           'label': label,
        #           # 'label_coef': label_coef,
        #           }
        sample = {'img': image, 'meta': meta, 'id': id, 'label': label, 'text': text, 'category': category,
                  'title': title}

        return sample


class youtube_data_lstm(Dataset):
    def __init__(self, csv_file, cover_dir, gt_path):
        self.data = pd.read_csv(csv_file)
        self.Hyfea = pd.read_csv("/home/zhuwei/SMP/1Ref/Hyfea/output0.csv")
        # self.data=self.data.drop(columns=['Day 2'])
        self.cover_dir = cover_dir
        # self.profile_dir = profile_diree

        self.cover_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        self.profile_transform = transforms.Compose([transforms.Resize([48, 48]), transforms.ToTensor()])

        self.visual_content = True  # False
        self.textual_content = True  # False#
        self.numerical_content = True  # False
        self.categorical_content = True
        self.p_i = 0
        self.seq_len = 29

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = str(self.data.iloc[idx, 0])

        if self.visual_content:
            cover_img_name = os.path.join(self.cover_dir, str(self.data.iloc[idx, 0]) + '.jpg')
            # profile_img_name = os.path.join(self.profile_dir, str(self.data.iloc[idx, 0]) + '.jpg')
            try:
                cover_img = self.cover_transform(Image.open(cover_img_name).convert("RGB"))
                # profile_img = self.profile_transform(Image.open(profile_img_name).convert("RGB"))
            except:
                cover_img = torch.zeros(3, 224, 224)
                # profile_img = torch.zeros(3, 224, 224)
                # print('no img:{}'.format(cover_img_name))
        else:
            cover_img = torch.zeros(3, 224, 224)
            # profile_img = torch.zeros(3, 224, 224)

        if self.textual_content:
            user_id = str(self.data.iloc[idx, 5])
            # user_id = "0"
            category = str(self.data.iloc[idx, 6])
            title = str(self.data.iloc[idx, 7])
            keyword = str(self.data.iloc[idx, 8])
            description = str(self.data.iloc[idx, 9])

        else:
            user_id = "0"
            category = "0"
            title = "0"
            keyword = "0"
            description = "0"

        text = [user_id, category, title, keyword, description]

        if self.categorical_content:
            category = str(self.data.iloc[idx, 6])
            title = str(self.data.iloc[idx, 7])
            # title = "0"
        else:
            category = "0"
            title = "0"

        cat = [category, title]

        if self.numerical_content:
            meta = torch.from_numpy(self.data.iloc[idx, 10:15].values.astype(np.float32))
            # meta[3:] = 0
        else:
            meta = torch.zeros(5)

        if self.p_i == 0:
            p_i = torch.zeros(1)
        elif self.p_i == 1:
            p_i = torch.from_numpy(self.data.iloc[idx, 15:16].values.astype(np.float32))
            # p_i = torch.zeros(3)
            # print(p_i)
        # elif self.p_i == 2:
        #     p_i = torch.from_numpy(self.data.iloc[idx, 16:19].values.astype(np.float32))
        meta = torch.cat([meta, p_i])

        if self.seq_len == 29:
            label_ = torch.from_numpy(self.data.iloc[idx, 16: 45].values.astype(np.float32))
        else:
            label_ = torch.from_numpy(self.data.iloc[idx, 15: 45].values.astype(np.float32))

        sample = {'img': cover_img,
                  'text': text,
                  # 'profile_img': profile_img,
                  'meta': meta,
                  'cat': cat,
                  'id': video_id,
                  'label': label_,
                  }

        return sample

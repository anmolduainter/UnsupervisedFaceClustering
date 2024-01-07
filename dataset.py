import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, root_dir, folder_l):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir

        datasetsl = folder_l

        images = []
        labels = []
        c = 0
        for dl in tqdm(datasetsl):
            rd = root_dir + dl + "/"
            classes = sorted(os.listdir(rd))
            for idx, class_name in enumerate(classes):
                c += 1
                class_dir = os.path.join(rd, class_name)
                if os.path.isdir(class_dir):
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        images.append(image_path)
                        labels.append(c)
                        # if (c == 30):
                        #     break

        print (c)
        print (len(images))
        self.image_paths = images
        self.labels = labels

    def denormalize(self, image_tensor):
        # Denormalize the image tensor
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = image_tensor.permute(1, 2, 0).numpy()  # Change tensor shape to HWC and convert to NumPy
        image = (image * std) + mean  # Denormalize
        image = np.clip(image, 0, 1)  # Clip values to stay within [0, 1] range
        return image
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112,112))
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(image)
        # image_np = self.denormalize(sample) 
        # image_np = (image_np * 255).astype(np.uint8)
        # cv2.imwrite('./img.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        return sample, label

    def __len__(self):
        return len(self.image_paths)
    
class EvaluationFaceDataset:
    def __init__(self, root_dir, folder_l):
        super(EvaluationFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        datasetsl = folder_l

        images = []
        labels = []
        for dl in datasetsl:
            rd = root_dir + dl + "/"
            classes = sorted(os.listdir(rd))
            for idx, class_name in enumerate(classes):
                class_dir = os.path.join(rd, class_name)
                c = 0
                if os.path.isdir(class_dir):
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        images.append(image_path)
                        labels.append(idx)
                        c += 1
                        # if (c == 30):
                        #     break

        
        self.image_paths = images
        self.labels = labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112,112))
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(image)
        return sample, label

    def __len__(self):
        return len(self.image_paths)
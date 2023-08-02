from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import numpy as np
import torch
import random
from torchvision.transforms import functional as TF

WIDTH = 256
HEIGHT = 256

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv.resize(image, (WIDTH, HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image/255.0
    return image

def read_mask(mask_path):
    image = cv2.imread(mask_path)
    # image = cv.resize(image, (WIDTH, HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask;
    red_mask[red_mask != 0] = 2

    # boundary Green color range values; Hue (36 - 70)
    green_mask = cv2.inRange(image, (36, 100, 20), (70, 255,255))
    green_mask[green_mask != 0] = 1

    full_mask = cv2.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask

def read_mask_bbam(path):
  img = cv2.imread(path, 0)
  img[img == 38] = 1
  img[img == 75] = 2

  return img

class Generator(Dataset):
    def __init__(self, mode, images_path, masks_path, augmentation_prob=0.5, model=None, len_data=0):
        self.images_path = images_path
        self.masks_path = masks_path
        self.mode = mode
        self.model = model
        self.augmentation_prob = augmentation_prob
        self.RotationDegree = [0,90,180,270]
        self.len_data = len_data

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        name = self.masks_path[idx].split("/")[-1].split(".")[0]
        if (self.model is None and idx < self.len_data):
          mask_path = self.masks_path[idx]                
          mask = read_mask(mask_path)
          one_hot_labels = torch.nn.functional.one_hot(torch.tensor(mask).long(), torch.tensor(3))
          mask = one_hot_labels.numpy()
        elif self.model is None and idx >= self.len_data:
          mask_path = f'data/pseudo_labels/{name}.png'
          mask = read_mask_bbam(mask_path)
          one_hot_labels = torch.nn.functional.one_hot(torch.tensor(mask).long(), torch.tensor(3))
          mask = one_hot_labels.numpy()
        else:
          image_path = f"data/WLIv5_pub_noud_640/Train/images/{name}.jpeg"
          image = read_image(image_path)
          image = Image.fromarray(np.uint8(image))
          Transform = []
          Transform.append(T.Resize((HEIGHT,WIDTH)))
          Transform.append(T.ToTensor())
          Transform = T.Compose(Transform)
          image = Transform(image)
          Transform =[]
          Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
          Transform = T.Compose(Transform)
          image = Transform(image)
          with torch.no_grad():
            mask = self.model(image.unsqueeze(0).cuda()).cpu()
          mask = mask.squeeze(0)
          return image, mask

        image_path = self.images_path[idx]
        image = read_image(image_path)

        image = Image.fromarray(np.uint8(image))
        mask = Image.fromarray(np.uint8(mask))

        Transform = []
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]

            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))

            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            mask = Transform(mask)

            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() < 0.3:
              Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

              image = Transform(image)

        if self.model is None:
          Transform =[]

          Transform.append(T.Resize((HEIGHT,WIDTH)))
          Transform.append(T.ToTensor())
          Transform = T.Compose(Transform)

          image = Transform(image)
          # image = image*255
          mask = Transform(mask)

          Transform =[]
          Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
          Transform = T.Compose(Transform)
          image = Transform(image)

          mask = torch.where(mask>0, torch.tensor(1), torch.tensor(0))

          return image, mask
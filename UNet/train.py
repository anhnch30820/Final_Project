from glob import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
# os.environ['CUDA_VISIBLE_DEVICES']="0"

dataset_path = "/home/s/anhnch/data/WLIv5_pub_noud_640/"                 
training_data = "Train/images/"                                    
training_mask = "Train/label_images/"  
test_data = "Test/images/"                                   
test_mask = "Test/label_images/"                                   

X_train = sorted(glob(dataset_path + training_data + "*.jpeg"))  
X_train_mask = sorted(glob(dataset_path + training_mask + "*.png"))    
X_test = sorted(glob(dataset_path + test_data + "*.jpeg"))    
X_test_mask = sorted(glob(dataset_path + test_mask + "*.png"))    
print(f"The Training Dataset contains {len(X_train)} images.")
print(f"The Training Dataset contains {len(X_train_mask)} images.")
print(f"The Testing Dataset contains {len(X_test)} images.")
print(f"The Testing Dataset contains {len(X_test_mask)} images.")

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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
from PIL import Image
import torch.nn.functional as F
# from keras.utils.np_utils import to_categorical

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
        if (self.model is None and idx < self.len_data) or self.mode == "test":
          mask_path = self.masks_path[idx]
          if self.mode == "test":
            # mask_path = self.masks_path[idx]
            mask_path = f"data/WLIv5_pub_noud_640/Test/label_images/{name}.png"       
          else:
            mask_path = f"data/WLIv5_pub_noud_640/Train/label_images/{name}.png"
          
          mask = read_mask(mask_path)
          one_hot_labels = torch.nn.functional.one_hot(torch.tensor(mask).long(), torch.tensor(3))
          mask = one_hot_labels.numpy()
          # mask = to_categorical(mask, num_classes=3)

        elif self.model is None and idx >= self.len_data:
        #   name = idx - 350
          mask_path = f'data/pseudo_labels_0.65_0.3/{name}.png'

          # mask = read_mask_semi(mask_path)
          mask = read_mask_bbam(mask_path)
          one_hot_labels = torch.nn.functional.one_hot(torch.tensor(mask).long(), torch.tensor(3))
          mask = one_hot_labels.numpy()
          # mask = to_categorical(mask, num_classes=3)

        else:
          # print("esle")
          image_path = f"data/WLIv5_pub_noud_640/Train/images/{name}.jpeg"
          # image_path = self.images_path[idx]
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
        if self.mode == "test":
          image_path = f"data/WLIv5_pub_noud_640/Test/images/{name}.jpeg"       
        else:
          image_path = f"data/WLIv5_pub_noud_640/Train/images/{name}.jpeg"
        image = read_image(image_path)

        # print(image.shape, mask.shape)

        image = Image.fromarray(np.uint8(image))
        mask = Image.fromarray(np.uint8(mask))
        # print(image.shape, mask.shape)

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

from sklearn.model_selection import train_test_split
# import cv2
# import matplotlib.pyplot as plt

# train_x_bbam, train_x_semi, train_y_bbam, train_y_semi = train_test_split(X_train, X_train_mask, test_size=0.5, random_state=42)


def reverse_transform(inp):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			  nn.ReLU(inplace=True)
        )


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        return x


class ATT(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(ATT,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class NeoUnet(nn.Module):
    def __init__(self):
        super(NeoUnet, self).__init__()
        self.model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
        children = list(self.model.children())
        self.conv1 = nn.Sequential(*children[0][0:2])
        self.conv2 = nn.Sequential(*children[0][2:5])
        self.conv3 = nn.Sequential(*children[0][5:10])
        self.conv4 = nn.Sequential(*children[0][10:13])
        self.conv5 = nn.Sequential(*children[0][13:16])



        self.Up5 = up_conv(ch_in=1024,ch_out=640)
        self.Att5 = ATT(F_g=640,F_l=640,F_int=320)
        self.Up_conv5 = conv_block(ch_in=1280, ch_out=640)

        self.Up4 = up_conv(ch_in=640,ch_out=320)
        self.Att4 = ATT(F_g=320,F_l=320,F_int=128)
        self.Up_conv4 = conv_block(ch_in=640, ch_out=320)

        self.Up3 = up_conv(ch_in=320,ch_out=128)
        self.Att3 = ATT(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = ATT(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64,ch_out=32)
        self.Att1 = ATT(F_g=32,F_l=3,F_int=3)
        self.Up_conv1 = conv_block(ch_in=35, ch_out=16)


        # self.Up_conv1 = conv_block(ch_in=35, ch_out=32)
        # self.Up_conv = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(16,3,kernel_size=1,stride=1,padding=0)


        # for param in self.conv1.parameters():
        #     param.requires_grad = False
        # for param in self.conv2.parameters():
        #     param.requires_grad = False
        # for param in self.conv3.parameters():
        #     param.requires_grad = False
        # for param in self.conv4.parameters():
        #     param.requires_grad = False
        # for param in self.conv5.parameters():
        #     param.requires_grad = False


    def forward(self, im_data):
        x1 = self.conv1(im_data)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)


        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4,x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3,x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2,x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x = self.Att1(d1,im_data)
        d1 = torch.cat((x,d1),dim=1)
        d1 = self.Up_conv1(d1)

        # d1 = self.Up1(d2)
        # d = torch.cat((im_data, d1),dim=1)
        # d = self.Up_conv1(d)
        # d = d+d1
        # d = self.Up_conv(d)

        ouput = (self.Conv_1x1(d1))

        return ouput

from copy import deepcopy

class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.3, device="cuda", epoch=0):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, epoch):  
        if epoch >=5:
          self.decay = 0.6
        if epoch >=15:
           self.decay = 0.9
        if epoch >=40:
           self.decay = 0.995

        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


########## IoU and Dice
import torch.nn.functional as F
def IoU_and_Dice_train(sample_mask, pred_mask):
    smooth = 0.001
    sample_mask1 = torch.argmax(sample_mask, axis=1)
    pred_mask1 = torch.argmax(pred_mask, axis=1)

    sample_mask1 = F.one_hot(sample_mask1, 3)
    pred_mask1 = F.one_hot(pred_mask1, 3)

    axes = (1, 2)
    intersection = torch.sum(torch.logical_and(pred_mask1, sample_mask1), axis=(axes))
    # intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    union = torch.sum(torch.logical_or(pred_mask1, sample_mask1), axis=axes)
    mask_sum = torch.sum(sample_mask1, axis=axes) + torch.sum(pred_mask1, axis=axes)
    # union = mask_sum  - intersection

    iou = (intersection + smooth) / (union + smooth)
    dice = (2*intersection + smooth)/(mask_sum + smooth)

    iou = torch.mean(iou)
    dice = torch.mean(dice)


    return iou, dice

def tversky(sample_mask, pred_mask):

    smooth = 0.001
    sample_mask1 = torch.argmax(sample_mask, axis=1)
    pred_mask1 = torch.argmax(pred_mask, axis=1)

    sample_mask1 = F.one_hot(sample_mask1, 3)
    pred_mask1 = F.one_hot(pred_mask1, 3)

    axes = (1, 2)
    true_pos = torch.sum(sample_mask1 * pred_mask1, axis=axes)
    false_neg = torch.sum(sample_mask1 * (1-pred_mask1), axis=axes)
    false_pos = torch.sum((1-sample_mask1)*pred_mask1)
    alpha = 0.3
    return torch.mean((true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth))

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 4/3
    return torch.pow((1-pt_1), gamma)

def filter_threshold(mask, pred):
  
  mask = torch.nn.functional.softmax(mask.float(), dim=1)
  pred = torch.nn.functional.softmax(pred, dim=1)
  valid_mask = (mask > 0.75) | (mask < 0.2)
  valid_mask_foreground = (mask > 0.75)
  valid_mask_background = (mask < 0.2)
  # print(mask[valid_mask_foreground])
  # print(len(mask[valid_mask_foreground]))
  # print(mask[valid_mask_background])
  # print(len(mask[valid_mask_background]))
  # Áp dụng mask để lấy các phần tử tương ứng trong mask và pred
  mask[valid_mask_foreground] = 1
  mask[valid_mask_background] = 0
  # print(mask[valid_mask_foreground])
  # print(mask[valid_mask_background])
  masked_mask = mask[valid_mask]
  # print(len(masked_mask))
  
  masked_pred = pred[valid_mask]


  return masked_mask, masked_pred



def train_case(model, train_dataloader, valid_dataloader, path_load_weights, path_best_weights_IoU, path_load_accuracy_and_loss, 
              plot_IoU,
              plot_val_IoU,
              plot_Dice,
              plot_val_Dice,
              plot_loss,
              plot_val_loss,
              valid_x_bbam,
              valid_y_bbam):

    ########## TRAIN MODEL
  # ##### PARAMETERS
  import math
  import torch.nn.functional as F
  from torch import optim
  epochs = 150
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion = nn.CrossEntropyLoss()
  bce_loss = nn.BCELoss()
  lr = 0.0001
  ##### TRAIN MODEL
  flag = True
  semi_dataloader = None
  train_semi_dataloader = None
  model_ema = None
  while len(plot_IoU) < epochs:

      model.train()

      cur_epoch = len(plot_IoU)
      epoch_semi = 555550
      epoch_valid = 0
      if cur_epoch >= epoch_valid and flag:
        flag = False
        model_ema = ModelEmaV2(model)
      if cur_epoch == epoch_semi and flag:
        flag = False
        model_ema = ModelEmaV2(model)
        semi_dataloader = Generator('valid', valid_x_bbam, valid_y_bbam, model=model_ema.module)
        train_semi_dataloader = DataLoader(semi_dataloader, batch_size=8, shuffle=True)
      if cur_epoch > epoch_semi:
        semi_dataloader = Generator('valid', valid_x_bbam, valid_y_bbam, model=model_ema.module)
        train_semi_dataloader = DataLoader(semi_dataloader, batch_size=8, shuffle=True)

      model.train()
      # if cur_epoch == 50:
      #   lr = lr*0.1
      optimizer = optim.Adam(model.parameters(), lr=float(lr))


      train_loss = 0
      loss = 0
      IoU = 0
      Dice = 0
      val_loss = 0
      val_IoU = 0
      val_Dice = 0
      length = 0
      for i in range(1):
        for (batch_idx, target_tuple) in enumerate(train_dataloader):
          
          # break
          images, masks = target_tuple

          images = images.to(device)
          masks = masks.to(device)

          optimizer.zero_grad()  # zero the gradient buff

          pred_masks = model(images)

          # loss = criterion(pred_masks, masks.float())
          # loss = bce_loss(torch.sigmoid(pred_masks), masks.float())
          loss = F.binary_cross_entropy_with_logits(pred_masks, masks.float())
          loss1 = focal_tversky(masks, pred_masks)

          iou, dice = IoU_and_Dice_train(masks, pred_masks)
          
          loss = loss*0.3 + ((1-iou)*0.5 + (1-dice)*0.5)*0.4 + loss1*0.3

          loss.backward()  # retain_graph=True
          optimizer.step()

          train_loss += loss.item()
          IoU += iou.item()
          Dice += dice.item()
          # length += images.size(0)

          if (batch_idx % 4 == 0 and cur_epoch>=epoch_semi) or (batch_idx % 4 == 0 and cur_epoch>=epoch_valid):
            # print(batch_idx)
            model_ema.update(model, epoch=cur_epoch)

          # print('########################### Epoch:', cur_epoch, ', --  batch:',  batch_idx, '/', len(train_dataloader))
          # print('average train loss: %.4f '%(train_loss/(batch_idx+1)))
          # print('average train IoU: %.4f '% (IoU/(batch_idx+1)))
          # print('average train Dice: %.4f '% (Dice/(batch_idx+1)))



        loss = train_loss/len(train_dataloader)
        IoU = IoU/len(train_dataloader)
        Dice = Dice/len(train_dataloader)
        print('average train loss after %d epoch: %.4f ' %(cur_epoch, loss))
        print('average train IoU after %d epoch: %.4f '%(cur_epoch, IoU))
        print('average train Dice after %d epoch: %.4f ' %(cur_epoch, Dice))

        ############ semi ############
        print("=================== SEMI ========================")
        train_loss_semi = 0
        IoU_semi = 0
        Dice_semi = 0
        if cur_epoch >= epoch_semi:
          for (batch_idx, images) in enumerate(train_semi_dataloader):
            images, masks = target_tuple

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()  # zero the gradient buff

            pred_masks = model(images)

            masks1, pred_masks1 = filter_threshold(masks, pred_masks)


            loss_semi = F.binary_cross_entropy(pred_masks1, masks1.float())
            # print(loss_semi)
        #     # loss1 = focal_tversky(masks, pred_masks)

        #     # iou, dice = IoU_and_Dice_train(masks, pred_masks)
        #     # loss_semi = (loss_semi*0.4 + (1-iou)*0.6)*0.6 + loss1*0.4

            loss_semi.backward()  # retain_graph=True
            optimizer.step()

            train_loss_semi += loss_semi.item()
            # IoU_semi += iou.item()
            # Dice_semi += dice.item()

            # if batch_idx % 32 == 0 :
            #   print(batch_idx)
            #   model_ema.update(model)

        #     print('########################### Epoch:', cur_epoch, ', --  batch:',  batch_idx, '/', len(train_semi_dataloader))
            # print('average train loss: %.4f '%(train_loss_semi/(batch_idx+1)))
        #     # print('average train IoU: %.4f '% (IoU_semi/(batch_idx+1)))
        #     # print('average train Dice: %.4f '% (Dice_semi/(batch_idx+1)))



          loss_semi = train_loss_semi/len(train_semi_dataloader)
        #   # IoU_semi = IoU_semi/len(train_semi_dataloader)
        #   # Dice_semi = Dice_semi/len(train_semi_dataloader)
          print('average train loss after %d epoch: %.4f ' %(cur_epoch, loss_semi))
        #   # print('average train IoU after %d epoch: %.4f '%(cur_epoch, IoU))
        #   # print('average train Dice after %d epoch: %.4f ' %(cur_epoch, Dice))



      print('\n ############################# Valid phase, Epoch: {} #############################'.format(cur_epoch))
      model.eval()

      with torch.no_grad():

        val_loss_no_grad = 0
        val_IoU_no_grad = 0
        val_Dice_no_grad = 0
        length = 0

        for (batch_idx, target_tuple) in enumerate(valid_dataloader):
          images, masks = target_tuple

          images = images.to(device)
          masks = masks.to(device)

          if cur_epoch >= epoch_semi or cur_epoch >= epoch_valid:
          # if False:
            pred_masks = model_ema.module(images)
          else:
            pred_masks = model(images)

          # val_loss = criterion(pred_masks, masks.float())
          # val_loss = bce_loss(torch.sigmoid(pred_masks), masks.float())
          val_loss = F.binary_cross_entropy_with_logits(pred_masks, masks.float())
          val_loss1 = focal_tversky(masks, pred_masks)

          iou, dice = IoU_and_Dice_train(masks, pred_masks)
          val_loss = val_loss*0.3 + ((1-iou)*0.5 + (1-dice)*0.5)*0.4 + val_loss1*0.3
          # val_loss = (val_loss*0.4 + (1-iou)*0.6)*0.6 + val_loss1*0.4

          val_loss_no_grad += val_loss.item()
          val_IoU_no_grad += iou.item()
          val_Dice_no_grad += dice.item()
          # length += images.size(0)



          # print('########################### Epoch:', cur_epoch, ', --  batch:',  batch_idx, '/', len(valid_dataloader))

          # print('average valid loss: %.4f '%(val_loss_no_grad/(batch_idx+1)))
          # print('average valid IoU: %.4f '% (val_IoU_no_grad/(batch_idx+1)))
          # print('average valid Dice: %.4f '% (val_Dice_no_grad/(batch_idx+1)))

        val_loss = val_loss_no_grad/len(valid_dataloader)
        val_IoU = val_IoU_no_grad/len(valid_dataloader)
        val_Dice = val_Dice_no_grad/len(valid_dataloader)

      print('average valid loss after %d epoch: %.4f ' %(cur_epoch, val_loss))
      print('average valid IoU after %d epoch: %.4f '%(cur_epoch, val_IoU))
      print('average valid Dice after %d epoch: %.4f ' %(cur_epoch, val_Dice))

      print("==================================")
      # show_predictions()                                ### callbacks function
      # print ('\nSample Prediction After Epoch {}\n'.format(len(plot_val_IoU+1)))   ### callbacks function

      torch.save(model.state_dict(), path_load_weights)       # Save data after training each epoch in case google colab is disconnected

      plot_IoU = np.append(plot_IoU,IoU)

      plot_val_IoU = np.append(plot_val_IoU,val_IoU)
      if val_IoU >= np.amax(plot_val_IoU):
        if cur_epoch >= epoch_semi or cur_epoch >= epoch_valid:
        # if False:
          torch.save(model_ema.module.state_dict(), path_best_weights_IoU)
        else:
          torch.save(model.state_dict(), path_best_weights_IoU)
      plot_Dice = np.append(plot_Dice,Dice)
      plot_val_Dice = np.append(plot_val_Dice,val_Dice)
      # if val_Dice >= np.amax(plot_val_Dice):
      #   torch.save(model.state_dict(), path_best_weights_Dice)
      plot_loss = np.append(plot_loss,loss)
      plot_val_loss = np.append(plot_val_loss,val_loss)

      dict2 = {'IoU': plot_IoU,
                'val_IoU': plot_val_IoU,
                'Dice': plot_Dice,
                'val_Dice': plot_val_Dice,
                'loss': plot_loss,
                'val_loss': plot_val_loss
              }
      accuracy_and_loss = pd.DataFrame(data = dict2)
      accuracy_and_loss.to_csv(path_load_accuracy_and_loss, index=False)

from pickle import load, dump     
with open("/home/s/anhnch/anhnch/Data_train.pkl", "rb") as decoded_pickle:
    train_x_bbam, train_x_semi, train_y_bbam, train_y_semi = load(decoded_pickle)

train_x, train_x_bbam, train_y, train_y_bbam = train_test_split(train_x_bbam, train_y_bbam, test_size=3/4, random_state=42)

a = (train_x, train_x_bbam, train_y, train_y_bbam)

from pickle import load, dump
with open("anhnch/Data_train_0.125.pkl", "wb") as decoded_pickle:
  dump(a, decoded_pickle)

case = [0.125]
for i in case:
  model = NeoUnet().cuda()

  from pickle import load, dump
  with open("anhnch/Data_train.pkl", "rb") as decoded_pickle:
    train_x_bbam, train_x_semi, train_y_bbam, train_y_semi = load(decoded_pickle)
  # train_x, train_x_bbam, train_y, train_y_bbam = train_test_split(train_x_bbam, train_y_bbam, test_size=i, random_state=42)
  with open(f"anhnch/Data_train_{i}.pkl", "rb") as decoded_pickle:   
    train_x, train_x_bbam, train_y, train_y_bbam = load(decoded_pickle)

  len_data = len(train_x)
  # training = Generator('train', train_x, train_y, len_data=len_data)
  # train_dataloader = DataLoader(training, batch_size=8, shuffle=True, num_workers=16)
  valid = Generator('test', X_test, X_test_mask)
  valid_dataloader = DataLoader(valid, batch_size=8, shuffle=False, num_workers=16)

  # type_model = f"weight_supervised_{i}"
  # import os
  # os.makedirs(f'{type_model}', exist_ok=True)
  # path_load_weights = f'{type_model}/model.pth'
  # path_best_weights_IoU = f'{type_model}/model_best_IoU.pth'

  # if os.path.isfile(path_load_weights):
	# # Load the pretrained Encoder
	#   model.load_state_dict(torch.load(path_load_weights))


  # import pandas as pd
  # path_load_accuracy_and_loss = f'{type_model}/model_accuracy_and_loss.csv'         # Load and save accuracy, validation accuracy, loss, and validation loss in case google colab is disconnected

  # if not os.path.isfile(path_load_accuracy_and_loss):
  #     dict1 = {'IoU': [],
  #             'val_IoU': [],
  #             'Dice': [],
  #             'val_Dice': [],
  #             'loss': [],
  #             'val_loss': []
  #             }
  #     accuracy_and_loss = pd.DataFrame(data = dict1)
  #     accuracy_and_loss.to_csv(path_load_accuracy_and_loss, index=False)
  # df = pd.read_csv(path_load_accuracy_and_loss)

  # plot_IoU = np.array(df.IoU)
  # plot_val_IoU = np.array(df.val_IoU)
  # plot_Dice = np.array(df.Dice)
  # plot_val_Dice = np.array(df.val_Dice)
  # plot_loss = np.array(df.loss)
  # plot_val_loss = np.array(df.val_loss)


  # train_case(model, train_dataloader, valid_dataloader, path_load_weights, path_best_weights_IoU, path_load_accuracy_and_loss, 
  #             plot_IoU,
  #             plot_val_IoU,
  #             plot_Dice,
  #             plot_val_Dice,
  #             plot_loss,
  #             plot_val_loss,
  #             train_x_semi,
  #             train_y_semi)

  model = NeoUnet().cuda()
  

  train_x.extend(train_x_bbam)
  train_y.extend(train_y_bbam)
  print(len(train_x))

  training = Generator('train', train_x, train_y, len_data=len_data)
  train_dataloader = DataLoader(training, batch_size=8, shuffle=True, num_workers=16)

  weakly_name = 1/2-i
  # type_model = f"weight_supervised_weakly_semi_{i}_{weakly_name}_0.5"
  type_model = f"weight_supervised_weakly_{i}_{weakly_name}_0.65_0.3"

  os.makedirs(f'{type_model}', exist_ok=True)
  path_load_weights = f'{type_model}/model.pth'
  path_best_weights_IoU = f'{type_model}/model_best_IoU.pth'

  if os.path.isfile(path_load_weights):
	# Load the pretrained Encoder
	  model.load_state_dict(torch.load(path_load_weights))


  import pandas as pd
  path_load_accuracy_and_loss = f'{type_model}/model_accuracy_and_loss.csv'         # Load and save accuracy, validation accuracy, loss, and validation loss in case google colab is disconnected

  if not os.path.isfile(path_load_accuracy_and_loss):
      dict1 = {'IoU': [],
              'val_IoU': [],
              'Dice': [],
              'val_Dice': [],
              'loss': [],
              'val_loss': []
              }
      accuracy_and_loss = pd.DataFrame(data = dict1)
      accuracy_and_loss.to_csv(path_load_accuracy_and_loss, index=False)
  df = pd.read_csv(path_load_accuracy_and_loss)

  plot_IoU = np.array(df.IoU)
  plot_val_IoU = np.array(df.val_IoU)
  plot_Dice = np.array(df.Dice)
  plot_val_Dice = np.array(df.val_Dice)
  plot_loss = np.array(df.loss)
  plot_val_loss = np.array(df.val_loss)

  train_case(model, train_dataloader, valid_dataloader, path_load_weights, path_best_weights_IoU, path_load_accuracy_and_loss, 
              plot_IoU,
              plot_val_IoU,
              plot_Dice,
              plot_val_Dice,
              plot_loss,
              plot_val_loss,
              train_x_semi,
              train_y_semi)












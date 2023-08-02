from glob import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd

from .model import Unet, ModelEmaV2
from .utils import *
from .generator import Generator
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

model = Unet().cuda()

##########  Type Ratio data #########
from sklearn.model_selection import train_test_split
train_x_bbam, train_x_semi, train_y_bbam, train_y_semi = train_test_split(X_train, X_train_mask, test_size=0.5, random_state=42)
train_x, train_x_bbam, train_y, train_y_bbam = train_test_split(train_x_bbam, train_y_bbam, test_size=3/4, random_state=42)

len_data = len(train_x)
train_x.extend(train_x_bbam)
train_y.extend(train_y_bbam)
training = Generator('train', train_x, train_y, len_data=len_data)
train_dataloader = DataLoader(training, batch_size=8, shuffle=True, num_workers=16)
valid = Generator('test', X_test, X_test_mask)
valid_dataloader = DataLoader(valid, batch_size=8, shuffle=False, num_workers=16)

size_supervised = 0.125
size_weakly = 0.5 - size_supervised
type_model = f"weight_supervised_weakly_semi_{size_supervised}_{size_weakly}_0.5"
import os
os.makedirs(f'{type_model}', exist_ok=True)
path_load_weights = f'{type_model}/model.pth'
path_best_weights_IoU = f'{type_model}/model_best_IoU.pth'

if os.path.isfile(path_load_weights):
  model.load_state_dict(torch.load(path_load_weights))

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


########### training ###################
epochs = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bce_loss = nn.BCELoss()
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=float(lr))

flag = True
semi_dataloader = None
train_semi_dataloader = None
model_ema = None
while len(plot_IoU) < epochs:

  cur_epoch = len(plot_IoU)
  epoch_semi = 0
  epoch_valid = 55555
  if cur_epoch >= epoch_valid and flag:
    flag = False
    model_ema = ModelEmaV2(model)
  if cur_epoch >= epoch_semi and flag:
    flag = False
    model_ema = ModelEmaV2(model)
    semi_dataloader = Generator('valid', train_x_semi, train_y_semi, model=model_ema.module)
    train_semi_dataloader = DataLoader(semi_dataloader, batch_size=8, shuffle=True)

  model.train()
  
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
      images, masks = target_tuple
      images = images.to(device)
      masks = masks.to(device)
      optimizer.zero_grad()  # zero the gradient buff
      pred_masks = model(images)

      loss = bce_loss(torch.sigmoid(pred_masks), masks.float())
      # loss = F.binary_cross_entropy_with_logits(pred_masks, masks.float())
      loss_focal_tversky   = focal_tversky(masks, pred_masks)
      iou, dice = IoU_and_Dice_train(masks, pred_masks)   
      loss = loss * 0.5 + loss_focal_tversky * 0.5
      # loss = loss * 0.3 + ((1-iou) * 0.5 + (1-dice) * 0.5) * 0.4 + loss_focal_tversky * 0.3

      loss.backward()  # retain_graph=True
      optimizer.step()

      train_loss += loss.item()
      IoU += iou.item()
      Dice += dice.item()

      if (batch_idx % 4 == 0 and cur_epoch>=epoch_semi) or (batch_idx % 4 == 0 and cur_epoch>=epoch_valid):
        model_ema.update(model, epoch=cur_epoch)

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
        loss_semi.backward()  # retain_graph=True
        optimizer.step()

        train_loss_semi += loss_semi.item()
      loss_semi = train_loss_semi/len(train_semi_dataloader)

      print('average train loss after %d epoch: %.4f ' %(cur_epoch, loss_semi))

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
          pred_masks = model_ema.module(images)
        else:
          pred_masks = model(images)

        val_loss = bce_loss(torch.sigmoid(pred_masks), masks.float())
        # val_loss = F.binary_cross_entropy_with_logits(pred_masks, masks.float())
        val_loss_focal_tversky = focal_tversky(masks, pred_masks)
        iou, dice = IoU_and_Dice_train(masks, pred_masks)
        # val_loss = val_loss*0.3 + ((1-iou)*0.5 + (1-dice)*0.5)*0.4 + val_loss_focal_tversky*0.3
        val_loss = val_loss*0.5 + val_loss_focal_tversky*0.5

        val_loss_no_grad += val_loss.item()
        val_IoU_no_grad += iou.item()
        val_Dice_no_grad += dice.item()

      val_loss = val_loss_no_grad/len(valid_dataloader)
      val_IoU = val_IoU_no_grad/len(valid_dataloader)
      val_Dice = val_Dice_no_grad/len(valid_dataloader)

    print('average valid loss after %d epoch: %.4f ' %(cur_epoch, val_loss))
    print('average valid IoU after %d epoch: %.4f '%(cur_epoch, val_IoU))
    print('average valid Dice after %d epoch: %.4f ' %(cur_epoch, val_Dice))

    print("==================================")

    ############# Save model ##################
    torch.save(model.state_dict(), path_load_weights)   
    plot_IoU = np.append(plot_IoU,IoU)
    plot_val_IoU = np.append(plot_val_IoU,val_IoU)
    if val_IoU >= np.amax(plot_val_IoU):
      if cur_epoch >= epoch_semi or cur_epoch >= epoch_valid:
        torch.save(model_ema.module.state_dict(), path_best_weights_IoU)
      else:
        torch.save(model.state_dict(), path_best_weights_IoU)
    plot_Dice = np.append(plot_Dice,Dice)
    plot_val_Dice = np.append(plot_val_Dice,val_Dice)
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
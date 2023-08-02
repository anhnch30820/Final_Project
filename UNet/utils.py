import torch.nn.functional as F
import torch

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
  valid_mask = (mask > 0.6)
  # Áp dụng mask để lấy các phần tử tương ứng trong mask và pred
  mask[valid_mask] = 1
  masked_mask = mask[valid_mask]  
  masked_pred = pred[valid_mask]

  return masked_mask, masked_pred


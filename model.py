# Implementation: https://www.kaggle.com/code/chekoduadarsh/pytorch-beginner-code-faster-rcnn
# Modified slightly for my usecase

from pathlib import Path
import sys
from typing import Optional, Union, Any
import pandas as pd
import numpy as np
import cv2
import json
import os
import re
import pydicom
import time
import warnings

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import random
paddingSize = 0

warnings.filterwarnings("ignore")


SAVE_DIR = Path("/scratch/rawhad/CSE507/practice_2/models/finetuning")
run_id = sys.argv[1]

DIR_INPUT = '/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'


df = pd.read_csv(f'{DIR_INPUT}/train.csv')
df.fillna(0, inplace=True)
df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0  # TODO: undertstand why?

# FasterRCNN handles class_id==0 as the background, but we have 14 as our class label for background .
df["class_id"] = df["class_id"] + 1
df.loc[df["class_id"] == 15, ["class_id"]] = 0

print("df Shape: "+str(df.shape))
print("No Of Classes: "+str(df["class_id"].nunique()))
df.sort_values(by='image_id').head(10)  # TODO: is it being shuffled later?


def label_to_name(id: int) -> str:
  labels: dict[int, str] = {
      0: "Aortic enlargement",
      1: "Atelectasis",
      2: "Calcification",
      3: "Cardiomegaly",
      4: "Consolidation",
      5: "ILD",
      6: "Infiltration",
      7: "Lung Opacity",
      8: "Nodule/Mass",
      9: "Other lesion",
      10: "Pleural effusion",
      11: "Pleural thickening",
      12: "Pneumothorax",
      13: "Pulmonary fibrosis"
  }
  return labels.get(id - 1, str(id))


image_ids: np.ndarray = df['image_id'].unique()
np.random.shuffle(image_ids)  # Shuffle the image IDs
TRAIN_SIZE = int(0.8*len(image_ids))
train_ids: np.ndarray = image_ids[:TRAIN_SIZE]  # Training split
valid_ids: np.ndarray = image_ids[TRAIN_SIZE:]  # Validation split
train_df = df[df['image_id'].isin(train_ids)]
valid_df = df[df['image_id'].isin(valid_ids)]
print('Num of training samples:', len(train_df))
print('Num of validation samples:', len(valid_df))

# Thanks -  https://www.kaggle.com/pestipeti/
DS_ITEM_TYPE = tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]


class VinBigDataset(Dataset):  # type: ignore
  def __init__(self, dataframe: pd.DataFrame, image_dir: str, transforms: Optional[Any] = None, stat: str = 'Train') -> None:
    super().__init__()
    self.image_ids = dataframe["image_id"].unique()
    self.df = dataframe
    self.image_dir = image_dir
    self.transforms = transforms
    self.stat = stat

  def __getitem__(self, index: int) -> DS_ITEM_TYPE:
    image_id = self.image_ids[index]
    records = self.df[(self.df['image_id'] == image_id)]
    records = records.reset_index(drop=True)
    dicom = pydicom.dcmread(f"{self.image_dir}/{image_id}.dicom")
    image = dicom.pixel_array

    # Photometric interpretation in DICOM images refers to the relationship between the pixel values and the actual intensity of the image.
    # It describes how the pixel values are related to the physical property being measured.
    # The `MONOCHROME1` tag indicates the image is stored with pixel values representing the minimum intensity (i.e., darker pixels have lower values).
    # This code inverts the image to represent pixel values as maximum intensity (i.e., darker pixels have higher values). `MONOCHROME2`
    if "PhotometricInterpretation" in dicom:
      if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image

    # rescale and normalize to range [0, 255]
    slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1.0
    intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0.0
    image = (image.astype(np.float32) * slope) + intercept  # y = mx + c
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = np.stack([image, image, image])
    image = image.transpose(1, 2, 0).astype(np.uint8)

    if self.stat == 'Train':
      # TODO: check by deleting this
      if records.loc[0, "class_id"] == 0:
        records = records.loc[[0], :]

      boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
      area = torch.as_tensor(area, dtype=torch.float32)
      labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

      # assume there is no crowd
      iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

      target = {}
      target['boxes'] = boxes
      target['labels'] = labels
      target['image_id'] = torch.tensor([index])
      target['area'] = area
      target['iscrowd'] = iscrowd

      if self.transforms:
        image = self.transforms(image)
      if target["boxes"].shape[0] == 0:
        # Albumentation cuts the target (class 14, 1x1px in the corner)
        target["boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 1.0, 1.0]]))
        target["area"] = torch.tensor([1.0], dtype=torch.float32)
        target["labels"] = torch.tensor([0], dtype=torch.int64)

      return image, target, image_ids

    else:
      if self.transforms:
        image = self.transforms(**sample)
      return image, image_id

  def __len__(self) -> int:
    return int(self.image_ids.shape[0])


def dilation(img: np.array) -> np.array:  # custom image processing function
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
  img = cv2.dilate(img, kernel, iterations=1)
  return img


class Dilation(ImageOnlyTransform):  # type: ignore
  def apply(self, img: np.array, **params) -> np.array: return dilation(img)  # type: ignore

import torch
from torchvision import transforms as T

def get_train_transform():
    return T.Compose([
        #T.RandomHorizontalFlip(p=0.5),
        #T.RandomVerticalFlip(p=0.5),
        #T.RandomRotation(degrees=45),
        #T.RandomResizedCrop(size=800, scale=(0.9, 1.1)),
        T.ToTensor(),
        # FasterRCNN will normalize.
        T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

def get_valid_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

def get_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])


# ===
# Model
# ===

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 15  # 14 Classes + 1 background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def collate_fn(batch: DS_ITEM_TYPE) -> tuple[DS_ITEM_TYPE]:
  return tuple(zip(*batch))


train_dataset = VinBigDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = VinBigDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()
# Create train and validate data loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Train dataset sample
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


class Averager:
  def __init__(self) -> None:
    self.current_total = 0.0
    self.iterations = 0.0

  def send(self, value: float) -> None:
    self.current_total += value
    self.iterations += 1

  @property
  def value(self) -> float:
    if self.iterations == 0:
      return 0.0
    else:
      return 1.0 * self.current_total / self.iterations

  def reset(self) -> None:
    self.current_total = 0.0
    self.iterations = 0.0


model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs = 2  # Low epoch to save GPU time


# Validation function
def validate_model(model: torch.nn.Module, valid_data_loader: DataLoader) -> tuple[float, float, float]:
  model.eval()
  val_loss_hist = Averager()
  val_metric = MeanAveragePrecision()

  with torch.no_grad():
    for images, targets, image_ids in valid_data_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      loss_dict = model(images, targets)
      outputs = model(images)
      val_metric.update(outputs, targets)

      losses = sum(loss for loss in loss_dict.values())
      loss_value = losses.item()
      val_loss_hist.send(loss_value)

  eval_metrics = val_metric.compute()
  val_map = eval_metrics['map'].item()
  val_iou = eval_metrics['map_50'].item()

  return val_loss_hist.value, val_map, val_iou


loss_hist = Averager()
train_loss_history = []
val_loss_history = []
val_map_history = []
val_iou_history = []
start = time.time()

for epoch in range(num_epochs):
  print(f"Epoch {epoch+1}/{num_epochs}")

  # Validate model at the start of each epoch
  val_loss, val_map, val_iou = validate_model(model, valid_data_loader)
  val_loss_history.append(val_loss)
  val_map_history.append(val_map)
  val_iou_history.append(val_iou)

  # Train loop
  model.train()
  loss_hist.reset()

  for images, targets, image_ids in train_data_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_hist.send(loss_value)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

  # Save the training metrics
  train_loss_history.append(loss_hist.value)

  # Update the learning rate
  if lr_scheduler is not None:
    lr_scheduler.step()

  print(f"Train Loss: {loss_hist.value: .4f}")
  print(f"Val Loss: {val_loss: .4f}, Val mAP: {val_map: .4f}, Val IoU: {val_iou: .4f}")

end = time.time()
print(f"Training completed in {(end - start) / 60: .2f} minutes")

# Save the training and validation metrics and loss history in a JSON file
metrics_dict = {
    'train_loss_history': train_loss_history,
    'val_loss_history': val_loss_history,
    'val_map_history': val_map_history,
    'val_iou_history': val_iou_history
}

# Save metrics
metrics_path = SAVE_DIR / f"run_{run_id}_training_metrics.json"
with open(metrics_path, 'w') as f:
  json.dump(metrics_dict, f)

# Save the final model
model_path = SAVE_DIR / f"run_{run_id}_faster_rcnn_final_model.pth"
torch.save(model.state_dict(), model_path)

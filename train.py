# Implementation: https://www.kaggle.com/code/chekoduadarsh/pytorch-beginner-code-faster-rcnn
# Modified slightly for my usecase

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
from typing import Optional, Union, Any
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import deque
import cv2
import glob
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore")


torch.set_float32_matmul_precision('high')
SAVE_DIR = Path("/scratch/rawhad/CSE507/practice_2/models/finetuning")
run_id = sys.argv[1]


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


class VinDrCXRDataLoaderLite():
  def __init__(self, root: str, split: str, batch_size: int, use_worker: bool = False, prefetch_size: int = 1, shuffle: bool = False):
    self.root = root
    self.batch_size = batch_size
    self.split = split
    self.shard_size = self._get_shard_size()
    self.curr_file_ptr = None
    self.shuffle = shuffle
    # labels
    self.df = pd.read_csv(f'{root}/train.csv')
    self.df.fillna(0, inplace=True)
    self.df.loc[self.df["class_id"] == 14, ['x_max', 'y_max']] = 1.0  # TODO: undertstand why?
    # FasterRCNN handles class_id==0 as the background, but we have 14 as our class label for background .
    self.df["class_id"] = self.df["class_id"] + 1
    self.df.loc[self.df["class_id"] == 15, ["class_id"]] = 0
    # metadata
    self.files = glob.glob(os.path.join(root, f'{split}_ds', f'shard_*.pkl'))
    with open(os.path.join(root, f'{split}_ds', f'shard_metadata.json'), 'r') as f:
      self.metadata_json = json.load(f)
    # multiprocessing
    self.use_worker = use_worker
    self.prefetch_size = prefetch_size
    self.workers: list[mp.Process] = []
    self.prefetch_thread = None
    self.offset = 0
    self.step = self.batch_size
    assert self.step <= self.shard_size, "Batch size * world size must be less than or equal to shard size"
    self.reset()

  def next_batch(self):
    if self.use_worker:
      return self._next_batch_from_queue()
    return self._next_batch()

  def _get_shard_size(self):
    with open(self.files[0], 'rb') as f:
      return len(pickle.load(f))

  def load_shard(self, file_ptr):
    with open(self.files[file_ptr], 'rb') as f:
      return pickle.load(f)

  def _next_batch(self):
    images: list[torch.Tensor] = [torch.from_numpy(x) for x in self.curr_shard[self.curr_idx:self.curr_idx+self.step]]
    self.curr_idx += self.step
    # drop last batch if it's smaller than batch_size
    if (self.curr_idx + self.step) >= len(self.curr_shard):
      self.curr_file_ptr = (self.curr_file_ptr + 1) % len(self.files)  # cycle through files if necessary
      self.curr_idx = self.offset
      self.curr_shard = self.load_shard(self.curr_file_ptr)
      if self.shuffle:
        random.shuffle(self.curr_shard)

    targets: list[dict[str, torch.Tensor]] = []
    image_ids = self.metadata_json[self.files[self.curr_file_ptr]][self.curr_idx: self.curr_idx + self.step]
    for id_ in image_ids:
      records = self.df[self.df['image_id'] == id_]
      boxes = torch.from_numpy(records[['x_min', 'y_min', 'x_max', 'y_max']].values)
      labels = torch.tensor(records["class_id"].values, dtype=torch.int64)
      targets.append(dict(boxes=boxes, labels=labels))
    return (images, targets)

  def reset(self):
    if not self.use_worker:
      if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
        self.curr_file_ptr = 0
        self.curr_shard = self.load_shard(self.curr_file_ptr)
        if self.shuffle:
          random.shuffle(self.curr_shard)
      self.curr_idx = self.offset

    else:
      # for efficiency reasons, we build a queue to prefetch batches
      self.prefetch_queue = mp.Queue(maxsize=self.prefetch_size)
      if len(self.workers) > 0:
        for worker in self.workers:
          worker.terminate()
          worker.close()
      self.workers = []
      worker = mp.Process(target=self.worker)
      worker.start()
      self.workers.append(worker)

  def worker(self):
    if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
      self.curr_file_ptr = 0
      self.curr_shard = self.load_shard(self.curr_file_ptr)
      if self.shuffle:
        random.shuffle(self.curr_shard)
    self.curr_idx = self.offset
    self._fill_queue()

  def _fill_queue(self):
    while True:
      # add a batch to the queue
      # multiprocessing
      if self.prefetch_queue.full():
        time.sleep(0.25)
      else:
        self.prefetch_queue.put(self._next_batch())

  def _next_batch_from_queue(self):
    # multiprocessing
    while self.prefetch_queue.empty():
      time.sleep(0.25)
    return self.prefetch_queue.get()


# Usage
device = 'cuda'
device_type = 'cuda'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 3e-4
SHARD_DIR = "/scratch/rawhad/CSE507/practice_2/preprocessed_shards"

train_loader = VinDrCXRDataLoaderLite(SHARD_DIR, 'train', batch_size=BATCH_SIZE,
                                      use_worker=True, prefetch_size=2, shuffle=True)
valid_loader = VinDrCXRDataLoaderLite(SHARD_DIR, 'valid', batch_size=BATCH_SIZE,
                                      use_worker=True, prefetch_size=2, shuffle=False)
train_len = train_loader.shard_size * len(train_loader.files)
valid_len = valid_loader.shard_size * len(valid_loader.files)

# setup model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# change the head
num_classes = 15  # 14 Classes + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

# Training loop
train_loss_history = []
val_map_history = []
val_iou_history = []
for epoch in range(NUM_EPOCHS):
  print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
  # eval
  model.eval()
  val_metric = MeanAveragePrecision()
  with torch.no_grad():
    for _ in range(valid_len // BATCH_SIZE):
      images = [image.to(device) for image in images]
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        outputs = model(images)
      val_metric.update(outputs, targets)
  eval_metrics = val_metric.compute()
  val_map_history.append(eval_metrics['map'].item())
  val_iou_history.append(eval_metrics['map_50'].item())
  # train
  model.train()
  epoch_loss = []
  for _ in range(train_len // BATCH_SIZE):
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      losses.backward()
    optimizer.step()
    epoch_loss.append(losses.item())
  train_loss_history.append(sum(epoch_loss)/len(epoch_loss))
  # log
  print(f"Train Loss: {train_loss_history[-1]: .4f}")
  print(f"Val mAP: {val_map_history[-1]: .4f}, Val IoU: {val_iou_history[-1]: .4f}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'run_{run_id}_trained_model.pth'))
with open(os.path.join(SAVE_DIR, f'run_{run_id}_train_history.json'), 'w') as f:
  json.dump({
      'train_loss': train_loss_history,
      'val_map': val_map_history,
      'val_iou': val_iou_history
  }, f)

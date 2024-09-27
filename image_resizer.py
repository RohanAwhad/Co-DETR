import cv2
import glob
import json
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

ROOT_DIR = '/scratch/rawhad/CSE507/practice_2/preprocessed_shards'
TRAIN_CSV = os.path.join(ROOT_DIR, 'train.csv')
TRAIN_SHARDS_DIR = os.path.join(ROOT_DIR, 'train_ds')
VALID_SHARDS_DIR = os.path.join(ROOT_DIR, 'valid_ds')
NEW_ROOT_DIR = '/scratch/rawhad/CSE507/practice_2/preprocessed_shards_2'
NEW_TRAIN_CSV = os.path.join(NEW_ROOT_DIR, 'train.csv')
NEW_TRAIN_SHARDS_DIR = os.path.join(NEW_ROOT_DIR, 'train_ds')
NEW_VALID_SHARDS_DIR = os.path.join(NEW_ROOT_DIR, 'valid_ds')
os.makedirs(NEW_TRAIN_SHARDS_DIR, exist_ok=True)
os.makedirs(NEW_VALID_SHARDS_DIR, exist_ok=True)


def main() -> None:
  df = pd.read_csv(TRAIN_CSV)
  df.fillna(0, inplace=True)
  df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
  # train shards
  shards_list = get_all_files(TRAIN_SHARDS_DIR)
  shards_metadata = get_shard_metadata(TRAIN_SHARDS_DIR)
  for shard_file in tqdm(shards_list, desc='Train Shards'):
    resized_imgs_list: list[np.ndarray] = []
    with open(shard_file, 'rb') as f:
      shard: list[np.ndarray] = pickle.load(f)
    for img, img_id in zip(shard, shards_metadata[shard_file]):
      h, w = img.shape[:2]
      records = df[df['image_id'] == img_id]
      records[['x_min', 'x_max']] = records[['x_min', 'x_max']] / w
      records[['y_min', 'y_max']] = records[['y_min', 'y_max']] / h
      resized_imgs_list.append(cv2.resize(img, (512, 512)))
    save_shard(resized_imgs_list, os.path.join(NEW_TRAIN_SHARDS_DIR, os.path.basename(shard_file)))
  # valid shards
  shards_list = get_all_files(VALID_SHARDS_DIR)
  shards_metadata = get_shard_metadata(VALID_SHARDS_DIR)
  for shard_file in tqdm(shards_list, desc='Validation Shards'):
    resized_imgs_list: list[np.ndarray] = []
    with open(shard_file, 'rb') as f:
      shard: list[np.ndarray] = pickle.load(f)
    for img, img_id in zip(shard, shards_metadata[shard_file]):
      h, w = img.shape[:2]
      records = df[df['image_id'] == img_id]
      records[['x_min', 'x_max']] = records[['x_min', 'x_max']] / w
      records[['y_min', 'y_max']] = records[['y_min', 'y_max']] / h
      resized_imgs_list.append(cv2.resize(img, (512, 512)))
    save_shard(resized_imgs_list, os.path.join(NEW_VALID_SHARDS_DIR, os.path.basename(shard_file)))
  df.to_csv(NEW_TRAIN_CSV)


def get_all_files(directory: str) -> list[str]:
  return glob.glob(os.path.join(directory, "*.pkl"))


def get_shard_metadata(directory: str) -> dict[str, list[str]]:
  with open(os.path.join(directory, 'shards_metadata.json'), 'r') as f:
    return json.load(f)


def save_shard(shard: list[np.ndarray], shard_file: str) -> None:
  with open(shard_file, 'wb') as f:
    pickle.dump(shard, f)


if __name__ == '__main__':
  main()

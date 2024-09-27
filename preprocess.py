import json
import multiprocessing as mp
import os
import numpy as np
import pickle
import pydicom
import random

from tqdm import tqdm
from typing import Tuple, List


def preprocess_dicom(image_path_list: list[str]) -> np.ndarray:
  images_list: list[np.array] = []
  for image_path in tqdm(image_path_list, desc='Processing'):
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array

    if "PhotometricInterpretation" in dicom:
      if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image

    slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1.0
    intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0.0
    image = (image.astype(np.float32) * slope) + intercept
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = np.stack([image, image, image])
    image = image.transpose(1, 2, 0).astype(np.uint8) / 255  # betwee [0, 1]
    images_list.append(image)
  return image_path_list, images_list


def save_shard(shard: List[np.ndarray], image_ids: List[str], shard_id: int, shard_dir: str) -> Tuple[List[str], str]:
  shard_path = os.path.join(shard_dir, f"shard_{shard_id}.pkl")
  with open(shard_path, 'wb') as f:
    pickle.dump(shard, f)
  return image_ids, shard_path


def preprocess_images(image_paths: list[str], output_dir: str, shard_size: int = 1000) -> None:
  os.makedirs(output_dir, exist_ok=True)
  shards: list[tuple[list[str], str]] = []
  shard: list[np.ndarray] = []
  image_ids: list[str] = []

  n_procs = 8
  print(f"Using {n_procs} processes")

  with mp.Pool(n_procs) as pool:
    chunksize = 100
    for img_paths, imgs in tqdm(pool.imap_unordered(preprocess_dicom, [image_paths[i:i+chunksize] for i in range(0, len(image_paths), chunksize)], chunksize=1), total=1+(len(image_paths)//chunksize)):
      shard.extend(imgs)
      image_ids.extend([os.path.splitext(os.path.basename(x))[0] for x in img_paths])

      if len(shard) == shard_size:
        shards.append(save_shard(shard, image_ids, len(shards), output_dir))
        shard = []
        image_ids = []

    if len(shard) > 0:
      shards.append(save_shard(shard, image_ids, len(shards), output_dir))

  # Save the shards metadata
  metadata: dict[str, list[str]] = {k: v for v, k in shards}
  with open(os.path.join(output_dir, "shards_metadata.json"), "w") as f:
    json.dump(metadata, f)


if __name__ == '__main__':
  # Usage
  DIR_TRAIN = f'/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/train'
  OUTPUT_DIR = "/scratch/rawhad/CSE507/practice_2/preprocessed_shards"
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  image_paths = [os.path.join(DIR_TRAIN, f) for f in os.listdir(DIR_TRAIN) if f.endswith(".dicom")]
  random.shuffle(image_paths)
  TRAIN_SIZE = int(0.8*len(image_paths))
  train_paths: list[str] = image_paths[:TRAIN_SIZE]  # Training split
  valid_paths: list[str] = image_paths[TRAIN_SIZE:]  # Validation split
  preprocess_images(train_paths, OUTPUT_DIR+'/train_ds', shard_size=1000)
  preprocess_images(valid_paths, OUTPUT_DIR+'/valid_ds', shard_size=1000)

import os
import pydicom
from PIL import Image
from typing import List

# Set the path to the folder containing DICOM images
DICOM_DIR: str = "/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/test"

# Set the path to the folder where PNG images will be saved
SAVE_DIR: str = "/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/test_pngs"

# Create the SAVE_DIR if it doesn't exist
if not os.path.exists(SAVE_DIR):
  os.makedirs(SAVE_DIR)

# Get a list of all files in the DICOM_DIR
files: List[str] = os.listdir(DICOM_DIR)

# Iterate over each file in the DICOM_DIR
for file in files:
  # Check if the file is a DICOM file
  if file.endswith(".dicom"):
    # Load the DICOM file
    dicom_file: str = os.path.join(DICOM_DIR, file)
    ds: pydicom.Dataset = pydicom.dcmread(dicom_file)

    # Get the pixel data from the DICOM file
    pixel_data: bytes = ds.pixel_array

    # Create an Image object from the pixel data
    img: Image = Image.fromarray(pixel_data)

    # Create the filename for the PNG image
    png_filename: str = os.path.splitext(file)[0] + ".png"

    # Save the PNG image to the SAVE_DIR
    png_path: str = os.path.join(SAVE_DIR, png_filename)
    img.save(png_path)

print("Conversion complete. PNG images saved at:", SAVE_DIR)

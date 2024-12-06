

import PIL.Image
import pandas as pd
import PIL
import os
import numpy as np

import json


def load_lexica_image_prompt_pairs(data_folder_pth, prms=None):

  pass

def load_diffusionDB_image_prompt_pairs(data_folder_pth, prms=None):
  """
  part 000001 contains 1000 image, subset to 100 using prms["subset_size"]
  """
  img_part_pth = "part-000001"
  # Open and read the JSON file
  with open(os.path.join(data_folder_pth, img_part_pth, "part-000001.json"), 'r') as file:
      data = json.load(file)

  prompt_img_pairs = []
  for i, (k, v) in enumerate(data.items()):
    if len(prompt_img_pairs) >= prms["subset_size"]:
      break

    # img_pth = os.path.join(data_folder_pth, img_part_pth, k)
    # if not os.path.exists(img_pth):
    #   print(f"image {k} not found, skipping")
    #   continue
    # img = PIL.Image.open(img_pth)
    id = k.split(".")[0]
    prompt = [v["p"]]  # diffusion db only has 1 prompt per image
    prompt_img_pairs.append((id, prompt, None))
  
  return prompt_img_pairs


def load_mscoco_image_prompt_pairs(data_folder_pth, prms=None):
  """

  Returns:
    prompt_img_pairs: list of tuples (coco_id, list(prompt), PIL.Image)
  """
  
  caps = pd.read_csv(os.path.join(data_folder_pth, "captions.csv"), header=0)
  coco_ids = caps["coco_id"].unique()

  prompt_img_pairs = []
  for id in coco_ids:
    img_pth = os.path.join(data_folder_pth, "images", f"{id}.png")
    img = PIL.Image.open(img_pth)  
    prompt = caps[caps["coco_id"] == id]["caption"].values.tolist()

    prompt_img_pairs.append((id, prompt, img))

  return prompt_img_pairs


if __name__ == "__main__":
  # test data loaders
  out = load_mscoco_image_prompt_pairs("../data/mscoco")
  print(out)
  out = load_diffusionDB_image_prompt_pairs("../data/diffusionDB", prms={"subset_size": 2})
  print(out)

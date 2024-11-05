

import PIL.Image
import pandas as pd
import PIL
import os



def load_lexica_image_prompt_pairs(data_folder_pth, prms=None):

  pass


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
  load_mscoco_image_prompt_pairs("../data/mscoco")

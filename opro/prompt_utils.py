# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""
import os
import torch
import torch.nn as nn
import time
from typing import Tuple, Dict
import PIL

# import google.generativeai as palm
import openai
import numpy as np
from os.path import expanduser
from urllib.request import urlretrieve
import torchvision.transforms as transforms
from torchmetrics.multimodal.clip_score import CLIPScore


def call_openai_server_single_prompt(
    prompt, client, model="gpt-4o-mini", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

  except Exception as e:
      print(e)
      time.sleep(5)
      return call_openai_server_single_prompt(prompt, client, model=model, max_decode_steps=max_decode_steps, temperature=temperature)


def call_openai_server_func(
    inputs, client, model="gpt-4o-mini", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        client,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )


def T2I(prompt, generator_pipe, device):
    # Uncomment if you are using CPU
    if device == "cpu":
        num_steps = 1
    else:
        # Uncomment if you are using GPU
        num_steps = 20

    generator = torch.Generator(device).manual_seed(0)
    image = generator_pipe(prompt, generator=generator, num_inference_steps=num_steps).images[0]
    return image


def relevance_scorer(prompt, image, device):
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    image_tensor = transform(image)
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    image_tensor = image_tensor.to(device)
    metric = metric.to(device)
    score = metric(image_tensor, prompt)
    # print("Relevance Score:", score.detach().round())
    return float(score.detach().round())


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def aesthetic_scorer(image, device):
    import clip
    model, preprocess = clip.load("ViT-L/14", device=device)
    aesthetic_model = get_aesthetic_model().to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return aesthetic_model(image_features.float()).detach().item()

def call_VLM_scorer(query, prompt, generator_pipe, device) -> Tuple[Dict[str, float], PIL.Image.Image]:
  #TODO: implement the VLM and scorer

  # T2I, currently--directly optimize rwritten query
  image = T2I(prompt, generator_pipe, device)

  # Constant relevance score
  relevance = relevance_scorer(query, image, device)
  aesthetic = aesthetic_scorer(image, device)

  # controlled score: such as aesthetic, final score is a combination of relevance and X

  scores = {"relevance": relevance, "aesthetics": aesthetic}

  return scores, image


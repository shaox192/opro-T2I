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
r"""The .py file for prompt optimization.

Usage:

Step 1: edit the starting instructions by modifying `initial_instructions`

Step 2: edit the training ratio by modifying `train_ratio`

Step 3: check if the model configs (like batch size) are the same as the actual serving configs

Step 4: run

```
python optimize_instructions.py \
    --optimizer="gpt-4o-mini" --scorer="text-bison" \
    --instruction_pos="A_begin" --dataset="gsm8k" --task="train"
```

The outputs will then be written to `outputs/optimization-results/` in the opro folder.

Notes:

1. One or more API keys may need to be provided:
- When using a Google-Cloud-served model (like text-bison at https://developers.generativeai.google/tutorials/text_quickstart), add `--palm_api_key=<your_key>`
- When using an OpenAI model, add `--openai_api_key=”<your_key>”`

2. The initial instructions should be provided in the "initial_instructions"
variable.
"""

import datetime
import functools
import os
import sys
import torch

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
# import google.generativeai as palm
import numpy as np
import openai
from opro import prompt_utils
from opro.optimization import opt_utils
from opro import data_utils
import pandas as pd

ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

_OPENAI_API_KEY = flags.DEFINE_string(
    "openai_api_key", "", "The OpenAI API key."
)

_OPENAI_API_BASE = flags.DEFINE_string(
    "openai_api_base", "", "The OpenAI API base."
)

_SCORER = flags.DEFINE_string(
    "scorer", "relevance", "metric of generated image scorer"
)

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "gpt-4o-mini", "The name of the optimizer LLM."
)

_DATASET = flags.DEFINE_string(
    "dataset", "diffusionDB", "The name of dataset to search for instructions on. [mscoco, diffusionDB]"
)

_PARAM_AGGREGATE_SCORES = flags.DEFINE_bool(
    "param-aggregate-scores", True, "whether to aggregate scores for each image, or separte relavance and other scores"
)

_PARAM_SUBSET_SIZE = flags.DEFINE_integer(
    "param-subset-size", 2, "how many images to use in the dataset (subsetting the 1k part-000001 diffusionDB dataset)"
)

_PARAM_NUM_SEARCH_STEPS = flags.DEFINE_integer(
    "param-num-search-steps", 3, "how many rounds of opro evolutions to run"
)

_PARAM_NUM_GEN_PER_SEARCH = flags.DEFINE_integer(
    "param-num-gen-per-search", 6, "number of prompts to be generated in each evolution step"
)


def main(_):
  openai_api_key = _OPENAI_API_KEY.value
  openai_api_base = _OPENAI_API_BASE.value
  scorer_name = _SCORER.value
  optimizer_llm_name = _OPTIMIZER.value
  dataset_name = _DATASET.value.lower()

  subset_size = _PARAM_SUBSET_SIZE.value
  aggregate_scores = _PARAM_AGGREGATE_SCORES.value
  num_generated_instructions_in_each_step = _PARAM_NUM_GEN_PER_SEARCH.value
  num_search_steps = _PARAM_NUM_SEARCH_STEPS.value

  assert dataset_name in [
    "mscoco",
    "diffusiondb",
  ], f"The dataset name:{dataset_name} must be one of [mscoco, diffusionDB]"

  assert scorer_name in {
      "relevance",
      "aesthetics",
  }
  assert optimizer_llm_name in {
      "gpt-3.5-turbo",
      "gpt-4",
      "gpt-4o-mini",
  }

  print(f"scorer: {scorer_name}, optimizer: {optimizer_llm_name}, dataset: {dataset_name}")

  # make sure the scorer and optimizer models are callable
  if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
    assert openai_api_key, "The OpenAI API key must be provided."
  else:
    raise NotImplementedError("only openai models for now")

  if dataset_name == "mscoco":
    root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "mscoco")
  elif dataset_name == "diffusiondb":
    root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "diffusionDB")
  else:
    raise NotImplementedError("only mscoco datasets for now")

  # =================== create the result directory ==========================
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      OPRO_ROOT_PATH,
      "outputs",
      "optimization-results",
      f"{dataset_name.upper()}-score-{scorer_name}-opt-{optimizer_llm_name}-{datetime_str}/",
  )
  result_by_image_folder = os.path.join(
      save_folder, "result_by_image"
  )
  os.makedirs(result_by_image_folder)
  print(f"result directory:\n{save_folder}")

  # ====================== scorer model configs ==============================
  # difference between num_decodes and batch_size:
  # - num_decodes: how many outputs we actually want for each input
  # - batch_size: the batch size in model serving, should equal to that in
  # model serving config

  #TODO VLM params
  scorer_prms = {}

  if scorer_name == "relevance":
    # scorer_prms[""] = ""
    pass
  elif scorer_name == "aesthetics":
    # scorer_prms[""] = ""
    pass
  else:
    raise NotImplementedError("only relevance and aesthetics scorers for now")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"--> Using device: {device} <--")

  call_scorer_server_func = functools.partial(
        prompt_utils.call_VLM_scorer,
        device = device,
      )

  # ====================== optimizer model configs ============================
  assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}

  client = openai.OpenAI(api_key=openai_api_key, base_url=openai_api_base)

  optimizer_gpt_max_decode_steps = 512
  optimizer_gpt_temperature = 1.0

  optimizer_llm_dict = dict()
  optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
  optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
  optimizer_llm_dict["batch_size"] = 1
  optimizer_llm_dict["num_decodes"] = 1
  call_optimizer_server_func = functools.partial(
      prompt_utils.call_openai_server_func,
      model=optimizer_llm_name,
      max_decode_steps=optimizer_gpt_max_decode_steps,
      temperature=optimizer_gpt_temperature,
  )

  # ====================== try calling the servers ============================
  print("\n======== testing the scorer and optimizer servers ===========")
  #TODO scorer test
  '''scorer_test_output = call_scorer_server_func(
      "orig_prompt_here", "img prompt here", "ground truth image here"
  )
  print(f"scorer test output scores: {scorer_test_output}")'''

  optimizer_test_output = call_optimizer_server_func(
      "Does the sun rise from the north? Just answer yes or no.",
      client=client,
      temperature=1.0,
  )
  print(f"number of optimizer output decodes: {len(optimizer_test_output)}")
  print(f"optimizer test output: {optimizer_test_output}")
  print("Finished testing the servers.")


  # ====================== read data ============================
  print("\n================ reading data in ==============")
  #TODO data loading
  if dataset_name == "mscoco":
    raw_data = data_utils.load_mscoco_image_prompt_pairs(root_data_folder_path)
  elif dataset_name == "diffusiondb":
    raw_data = data_utils.load_diffusionDB_image_prompt_pairs(root_data_folder_path, prms={"subset_size": subset_size})

  else:
    raise NotImplementedError("only mscoco, diffusionDB datasets for now")

  num_examples = len(raw_data)
  print(f"number of images: {num_examples}, number of initial prompts for image 1: {len(raw_data[0][1])}")


  # ========== set other optimization experiment hyperparameters ==============

  #TODO: set parameters for scorer
  if scorer_name == "relevance":
    old_instruction_score_threshold = 0.3
  elif scorer_name == "aesthetics":
    old_instruction_score_threshold = 0.3

  optimizer_llm_temperature = optimizer_llm_dict["temperature"]

  max_num_instructions = 20  # the maximum number of instructions and scores in the meta-prompt

  # The number of buckets when converting scores to integers in the meta-prompt.
  # num_score_buckets = 100

  # ===================== run prompt optimization ======================
  evolution_kwargs = {
      "llm_client": client,
      "num_search_steps": num_search_steps,
      "old_instruction_score_threshold": old_instruction_score_threshold,
      "optimizer_llm_dict": optimizer_llm_dict,
      "num_examples": num_examples,
      "root_data_folder_path": root_data_folder_path,
      "optimizer_llm_temperature": optimizer_llm_temperature,
      "raw_data": raw_data,
      "call_scorer_server_func": call_scorer_server_func,
      "call_optimizer_server_func": call_optimizer_server_func,
      "max_num_instructions": max_num_instructions,
      "optimizer_llm_name": optimizer_llm_name,
      "num_generated_instructions_in_each_step": (
          num_generated_instructions_in_each_step
      ),
      "save_folder": save_folder,
      "result_by_image_folder":result_by_image_folder,
      "aggregate_scores": aggregate_scores,
  }

  opt_utils.run_evolution_T2I(**evolution_kwargs)


if __name__ == "__main__":
  app.run(main)

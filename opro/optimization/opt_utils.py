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
"""The utility functions for prompt optimization."""

import collections
import json
import os
import pickle as pkl
import re
import sys
from io import BytesIO

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import numpy as np
from opro.evaluation import eval_utils
import pandas as pd
# import openai
import time
from opro import prompt_utils


def extract_string_in_square_brackets(input_string):
  raw_result = re.findall(r"\[.*?\]", input_string)
  if raw_result:
    return raw_result[0][1:-1]
  else:
    return ""


def parse_tag_content(text, prefix="<TEXT>", suffix="</TEXT>"):
  pattern = f"{prefix}(.*?){suffix}"
  results = re.findall(pattern, text, re.DOTALL)
  return results


def _bucketize_float(num, n_buckets=20):
  assert num >= 0 and num <= 1, "The given number must be between 0 and 1."
  return round(num * n_buckets)


def gen_ins_and_score_pairs_substr(
    old_instructions_and_scores,
    aggregate_scores=True,
    old_instruction_score_threshold=0.1,
    max_num_instructions=1000,
    return_str_only=False,
    num_score_buckets=np.inf,
):
  """Generate the string that includes instruction-score pairs."""
  assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)
  old_instructions_and_scores_str = ""

  if aggregate_scores:
    old_instructions_and_scores = sorted(
        old_instructions_and_scores, key=lambda x: x[1]["total"]
    )[-max_num_instructions:]
  else:
    old_instructions_and_scores = old_instructions_and_scores[-max_num_instructions:]

  old_instructions_and_scores_in_meta_prompt = []
  old_byte_imgs = []
  for instruction, score_dict, i_step, byte_imgs in old_instructions_and_scores:
    old_byte_imgs.append(byte_imgs)
    if aggregate_scores:
      score = score_dict["total"]
      if (
          not old_instruction_score_threshold
          or score >= old_instruction_score_threshold
      ):
        old_instructions_and_scores_in_meta_prompt.append(
            (instruction, score, i_step)
        )
        if num_score_buckets == np.inf:
          score_to_show = round(score, 3)
        else:
          score_to_show = _bucketize_float(score, num_score_buckets)
        old_instructions_and_scores_str += (
            f"\ntext:\n{instruction}\n\nscore:\n{score_to_show}\n"
        )
    else:
      rel_score = score_dict["relevance"]
      aes_score = score_dict["aesthetics"]

      old_instructions_and_scores_in_meta_prompt.append(
            (instruction, (rel_score, aes_score), i_step)
        )
      
      if num_score_buckets == np.inf:
          rel_score_to_show = round(rel_score, 3)
          aes_score_to_show = round(aes_score, 3)
      else:
          rel_score_to_show = _bucketize_float(rel_score, num_score_buckets)
          aes_score_to_show = _bucketize_float(aes_score, num_score_buckets)
      old_instructions_and_scores_str += (
            f"\ntext:\n{instruction}\n\nrelevance score:\n {rel_score_to_show}\naesthetics score:\n {aes_score_to_show}\n"
        )
      
  if return_str_only:
    return (
      old_instructions_and_scores_str, 
      old_byte_imgs,
    )
  else:
    return (
        old_instructions_and_scores_str,
        old_instructions_and_scores_in_meta_prompt,
        old_byte_imgs
    )


META_INSTRUCTION = [
  [  # aggregate scores, text only
    "I have some texts along with their corresponding scores."
    " The texts are arranged in ascending order based on their scores,"
    " where higher scores indicate better quality.\n\n", 

    "\n\nWrite your new text that is different from the old ones and"
    " has a score as high as possible. Avoid overly long or overly short texts. "
    "Write the text in square brackets."
  ],
  [  # separate scores, text only
    "I have some texts, each accompanied by two scores: a relevance score and an aesthetics score."
    " Higher scores indicate better quality for their respective criteria. \n\n", 
    
   "\n\nWrite your new text that is different from the old ones to"
    " maximize both the relevance score and the aesthetics score simultaneously. "
    "Avoid overly long or overly short texts. "
    "Write the text in square brackets."
  ],
  [ # aggregate scores, text and image
    "I have some texts, their corresponding images, and associated scores. "
    "Each text is used to generate its corresponding image, and the scores reflect the quality of the generated image. "
    "The texts are arranged in ascending order based on their scores. \n\n",

    "\n\nWrite your new text that is different from the old ones and "
    "has a score as high as possible. Avoid overly long or overly short texts. "
    "Write the text in square brackets."
  ],
  [ # aggregate scores, text and image
    "I have some texts, their corresponding images, and two associated scores: two scores: a relevance score and an aesthetics score."
    "Each text is used to generate its corresponding image, "
    "and the scores reflect the quality of the generated image for their respective criteria. \n\n", 

    "\n\nWrite your new text that is different from the old ones to"
    " maximize both the relevance score and the aesthetics score simultaneously. "
    "Avoid overly long or overly short texts. "
    "Write the text in square brackets."
  ]
]
  

def gen_meta_prompt_T2I(
    old_instructions_and_scores,
    aggregate_scores=True,
    multi_modal=False,
    old_instruction_score_threshold=0.1,
    max_num_instructions=1000,
    num_score_buckets=np.inf,
):
  """Generate meta prompt for instruction rewriting.

  Args:
   old_instructions_and_scores (list): a list of (instruction, score, i_step)
     pairs.
   aggregate_scores (bool): whether to aggregate scores.

   old_instruction_score_threshold (float): only add old instructions with score
     no less than this threshold.
   max_num_instructions (int): the maximum number of instructions in the meta
     prompt.
   num_score_buckets (np.inf or int): the number of score buckets when we
     convert float accuracies to integers. Default to np.inf for not
     bucketizing.

  Returns:
   meta_prompt (str): the generated meta prompt.
  """
  # add old instructions
  old_instructions_and_scores_str, byte_imgs = gen_ins_and_score_pairs_substr(
      old_instructions_and_scores=old_instructions_and_scores,
      aggregate_scores=aggregate_scores,
      old_instruction_score_threshold=old_instruction_score_threshold,
      max_num_instructions=max_num_instructions,
      return_str_only=True,
      # num_score_buckets=num_score_buckets,
  )
  #TODO: redesign the meta instructions
  if multi_modal:
    if aggregate_scores:
      meta_ins1, meta_ins2 = META_INSTRUCTION[2]
    else:
      meta_ins1, meta_ins2 = META_INSTRUCTION[3]
  else:
    if aggregate_scores:
      meta_ins1, meta_ins2 = META_INSTRUCTION[0]
    else:
      meta_ins1, meta_ins2 = META_INSTRUCTION[1]

  meta_prompt = meta_ins1 + old_instructions_and_scores_str + meta_ins2

  return meta_prompt, byte_imgs


def eval_prompts(orig_query, prompts_ls, gt_img, scorer, verbose=False, step=-1):
  """
  
  orig_query: original query
  prompts_ls: list of re-written prompts
  Returns:
    scores (list): the scores of the prompts.
  """

  # evaluate initial instructions
  scores = []
  gen_img_ls = []
  for i, pro in enumerate(prompts_ls):
    if verbose:
      print(f"computing the score of '{pro}' by prompting")

    score_dict, img = scorer(orig_query, pro)  # TODO: query as the first parameter
    scores.append(score_dict)
    gen_img_ls.append(img)

  # average_score = np.average(scores)
  # print(f"average score: {average_score}")

  return scores, gen_img_ls


def calc_aggregated_score(scores_dict):
  tot_score = (scores_dict["relevance"] / 4 + scores_dict["aesthetics"]) / 2
  tot_score = np.exp(0.7 * tot_score)
  return tot_score

def save_img(img_ls, step_i, img_id, save_dir):
  byte_imgs_ls = []
  for img in img_ls:
    # width, height = img.size
    # print(f"Image dimensions: {width} x {height}")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    # file_size = len(buffer.getvalue())
    # print(f"Image size: {file_size / 1024:.2f} KB")  # Convert to kilobytes
    # exit()
    byte_img = buffer.getvalue()
    byte_imgs_ls.append(byte_img)

  save_f = os.path.join(save_dir, f"{img_id}-step-{step_i}.pkl")
  print(f"*** Saving images at step {step_i} to {save_f}")

  pkl.dump(byte_imgs_ls, open(save_f, "wb"))


def run_evolution_T2I(**kwargs):
  """The function for evolution."""
  # ================= experiment configurations =============================
  raw_data = kwargs["raw_data"]
  llm_client = kwargs["llm_client"]

  num_search_steps = kwargs["num_search_steps"]
  old_instruction_score_threshold = kwargs["old_instruction_score_threshold"]
  optimizer_llm_dict = kwargs["optimizer_llm_dict"]

  optimizer_llm_temperature = kwargs["optimizer_llm_temperature"]
  optimizer_llm_temperature_schedule = (
      kwargs["optimizer_llm_temperature_schedule"]
      if "optimizer_llm_temperature_schedule" in kwargs
      else "constant"
  )
  optimizer_llm_temperature_end = (
      kwargs["optimizer_llm_temperature_end"]
      if "optimizer_llm_temperature_end" in kwargs
      else None
  )

  call_scorer_server_func = kwargs["call_scorer_server_func"]
  call_optimizer_server_func = kwargs["call_optimizer_server_func"]

  # num_score_buckets = kwargs["num_score_buckets"]
  max_num_instructions = kwargs["max_num_instructions"]

  num_generated_instructions_in_each_step = kwargs[
      "num_generated_instructions_in_each_step"
  ]

  save_folder = kwargs["save_folder"]
  result_by_image_folder = kwargs["result_by_image_folder"]
  verbose = kwargs["verbose"] if "verbose" in kwargs else False
  aggregate_scores = kwargs["aggregate_scores"] if "aggregate_scores" in kwargs else True
  multi_modal = kwargs["multi_modal"] if "multi_modal" in kwargs else False

  # =================== assertions =====================
  assert optimizer_llm_temperature_schedule in {
      "constant",
      "linear_increase",
  }, "The temperature schedule should be constant or linear_increase."

  # =================== save configurations to json file ====================
  configs_dict = dict()
  configs_dict["optimizer_llm_dict"] = optimizer_llm_dict
  configs_dict["optimizer_llm_temperature"] = optimizer_llm_temperature
  configs_dict["optimizer_llm_temperature_schedule"] = (
      optimizer_llm_temperature_schedule
  )
  configs_dict["optimizer_llm_temperature_end"] = optimizer_llm_temperature_end
  with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
    json.dump(configs_dict, f, indent=4)
  # ===================================================================== 

  print(
      f"optimizer llm temperature: {optimizer_llm_temperature}, schedule:"
      f" {optimizer_llm_temperature_schedule}"
  )
  print(
      f"generating {num_generated_instructions_in_each_step} instructions in"
      f" each step, run for {num_search_steps} steps"
  )
  print(
      "discarding generated instructions with score less than:"
      f" {old_instruction_score_threshold} (old_instruction_score_threshold)",
  flush=True)
  # print(f"num_score_buckets: {num_score_buckets}")


  #TODO: maybe parallelize this?
  for i, (im_id, prompt_ls, img) in enumerate(raw_data):
    
    ### some cumulative parameters
    # the new instructions, format: [(instruction, score, step_index)]
    old_instructions_and_scores = []
    # meta_prompts = []  # format: [(meta_prompt, step_index)]
    # old_instruction_md5_hashstrings_set = set() # to avoid re-evaluating instructions
    ###

    print("\n============== evaluating initial instructions ===============")
    tik = time.time()
    score_ls, bl_gen_img_ls = eval_prompts(prompt_ls[0], prompt_ls, img, call_scorer_server_func, verbose, -1)
    print(f"Time taken for initial evaluation {len(prompt_ls)} prompts: {time.time() - tik :.3f} seconds", flush=True)

    save_img(bl_gen_img_ls, -1, im_id, result_by_image_folder)

    for j, p in enumerate(prompt_ls):
      curr_sc = score_ls[j]
      if aggregate_scores:
        curr_sc["total"] = calc_aggregated_score(curr_sc)

      byte_img = prompt_utils.encode_image(bl_gen_img_ls[j]) if multi_modal else None
      old_instructions_and_scores.append((p, curr_sc, -1, byte_img))

    # print(old_instructions_and_scores)

    # evolution
    print("\n============== Optimizing ===============")
    for i_step in range(num_search_steps):
      print(f"\n--Step {i_step + 1}/{num_search_steps}--")
      # if not i_step % 10:
      #   print(f"*old_instructions_and_scores: {old_instructions_and_scores}")

      if optimizer_llm_temperature_schedule == "linear_increase":
        optimizer_llm_temperature_curr = (
            optimizer_llm_temperature
            + i_step
            / num_search_steps
            * (optimizer_llm_temperature_end - optimizer_llm_temperature)
        )
      else:
        optimizer_llm_temperature_curr = optimizer_llm_temperature
      print(f"*current optimizer_llm_temperature: {optimizer_llm_temperature_curr}")

      # generate new instructions
      meta_prompt, byte_imgs = gen_meta_prompt_T2I(
          old_instructions_and_scores=old_instructions_and_scores,
          aggregate_scores=aggregate_scores,
          multi_modal=multi_modal,
          old_instruction_score_threshold=old_instruction_score_threshold,
          max_num_instructions=max_num_instructions,
      )
      
      if not i_step % 2:
        print("\n**************************************************")
        print(f"*meta_prompt: \n\n{meta_prompt}\n")
        print("**************************************************\n", flush=True)

      # meta_prompts.append((meta_prompt, i_step))
      remaining_num_instructions_to_generate = num_generated_instructions_in_each_step
      generated_instructions_raw = []
      # client = openai.OpenAI(api_key="<your_openai_api_key>", base_url="<your_openai_api_base>")
      tik = time.time()
      if not multi_modal:
        optimizer_llm_input_text = meta_prompt
      else:
        optimizer_llm_input_text = [
            {
              "type": "text",
              "text": meta_prompt,
            }
        ]
        optimizer_llm_input_text += [
            {
              "type": "image_url",
              "image_url": {
                "url":  f"data:image/jpeg;base64,{byte_im}"
              },
            } 
            for byte_im in byte_imgs
        ]
        
        optimizer_llm_input_text = [optimizer_llm_input_text]
      
      while remaining_num_instructions_to_generate > 0:
        # generate instructions
        # print(f"current temperature: {optimizer_llm_temperature_curr}")
        raw_outputs = call_optimizer_server_func(
            optimizer_llm_input_text,
            client = llm_client,
            temperature=optimizer_llm_temperature_curr,
        )

        # Extract the generated instructions from the optimizer LLM output. Only
        # keep some samples if the desired number of remaining instructions
        # is smaller than the total number of decodes in this step.
        raw_outputs = raw_outputs[:remaining_num_instructions_to_generate]
        generated_instructions_raw += [
            extract_string_in_square_brackets(string)
            for string in raw_outputs
        ]
        remaining_num_instructions_to_generate -= optimizer_llm_dict[
            "batch_size"
        ]

      generated_instructions_raw = list(
          map(eval_utils.polish_sentence, generated_instructions_raw)
      )
      print(f"\n * generated {len(generated_instructions_raw)} instructions in {time.time() - tik :.3f} seconds", flush=True)

      # do not evaluate old instructions again
      generated_instructions = []  # the new instructions generated in this step
      for ins in generated_instructions_raw:
        generated_instructions.append(ins)
        # ins_md5_hashstring = eval_utils.instruction_to_filename(
        #     ins, md5_hashing=True
        # )
        # if ins_md5_hashstring not in old_instruction_md5_hashstrings_set:
        #   generated_instructions.append(ins)
        #   old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
        # else:
        #   print(f"already evaluated '{ins}' previously")
      generated_instructions = list(set(generated_instructions))
      # print(generated_instructions)
      
      # get rid of instructions that are too long
      to_evaluate_instructions = []
      for iii, instruction in enumerate(generated_instructions):
        if instruction.startswith("Text: " or "text: "):
          instruction = instruction[len("Text: "):]

        if len(instruction) > 1000:
          print(f"Step {i_step}, {iii}-th instruction, too long, skipped")
          continue
        
        to_evaluate_instructions.append(instruction)
      print(f"\nnumber of to-evaluate generated instructions: {len(to_evaluate_instructions)}\n")
      if len(to_evaluate_instructions) == 0:
        continue

      # evaluate these newly generated prompts: 
      orig_query_ls = [triplet[0] for triplet in old_instructions_and_scores if triplet[2] == -1]
      score_ls, opt_gen_img_ls = eval_prompts(orig_query_ls[0], to_evaluate_instructions, img, call_scorer_server_func, verbose, i_step)
      save_img(opt_gen_img_ls, i_step, im_id, result_by_image_folder)

      if aggregate_scores:
        cum_sc = 0
        for ii, sc in enumerate(score_ls):
          score_ls[ii]["total"] = calc_aggregated_score(sc)
          cum_sc += score_ls[ii]["total"]
        average_score = cum_sc / len(score_ls)
        print(f"Step {i_step}, avg_score: {average_score}")
      
      else:
        cum_rel, cum_aes = 0, 0
        for ii, sc in enumerate(score_ls):
          cum_rel+= sc["relevance"]
          cum_aes+= sc["aesthetics"]
        print(f"Step {i_step}, relevance: {cum_rel/len(score_ls)}, aesthetics: {cum_aes/len(score_ls)}")      

      # save this step
      for j, p in enumerate(to_evaluate_instructions):
        byte_img = prompt_utils.encode_image(opt_gen_img_ls[j]) if multi_modal else None
        old_instructions_and_scores.append((p, score_ls[j], i_step, byte_img))

      # ===================== save up-to-date results ===========================
      results = {}
      results["instructions"] = [d[0] for d in old_instructions_and_scores]
      results["scores-rel"] = [d[1]["relevance"] for d in old_instructions_and_scores]
      results["scores-aes"] = [d[1]["aesthetics"] for d in old_instructions_and_scores]
      if aggregate_scores:
        results["scores-total"] = [d[1]["total"] for d in old_instructions_and_scores]
      results["step"] = [d[2] for d in old_instructions_and_scores]
      results = pd.DataFrame(results)
      save_p = os.path.join(result_by_image_folder, f"{im_id}.csv")
      results.to_csv(save_p, index=False)
      print(f"saved results for image: {im_id} to: {save_p}")


if __name__ == "__main__":
  # 1. test gen_meta_prompt_T2I
  old_instructions_and_scores = [
    ("This is a test instruction 1", 0.1, 0),
    ("This is a test instruction 2", 0.2, 1),
    ("This is a test instruction 3", 0.3, 2),
  ]

  meta_prompt = gen_meta_prompt_T2I(
    old_instructions_and_scores=old_instructions_and_scores,
    old_instruction_score_threshold=0.1,
    max_num_instructions=5
  )

  print(meta_prompt)
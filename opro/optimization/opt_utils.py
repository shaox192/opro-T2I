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
  for instruction, score_dict, i_step in old_instructions_and_scores:
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
    return old_instructions_and_scores_str
  else:
    return (
        old_instructions_and_scores_str,
        old_instructions_and_scores_in_meta_prompt,
    )


# def gen_meta_prompt(
#     old_instructions_and_scores,
#     instruction_pos,
#     optimizer_llm_name,
#     old_instruction_score_threshold=0.1,
#     max_num_instructions=1000,
#     meta_prompt_type="both_instructions_and_exemplars",
#     few_shot_qa_pairs=False,
#     include_qa=True,
#     data=None,
#     few_shot_index_list=None,
#     instructions_before_exemplars=True,
#     num_score_buckets=np.inf,
#     dataset_name="",
#     task_name="",
# ):
#   """Generate meta prompt for instruction rewriting.

#   Args:
#    old_instructions_and_scores (list): a list of (instruction, score, i_step)
#      pairs.
#    instruction_pos (str): where to put the instruction, one of {'before_QA',
#      'Q_begin', 'Q_end', 'A_begin'}.
#    optimizer_llm_name (str): the name of the LLM used for instruction editing.
#    old_instruction_score_threshold (float): only add old instructions with score
#      no less than this threshold.
#    max_num_instructions (int): the maximum number of instructions in the meta
#      prompt.
#    meta_prompt_type (str): the type of meta-prompt: whether to have both
#      previous instructions and dataset exemplars (often for fine-tuned
#      optimizers), or to have only previous instructions (often for pre-trained
#      optimizers).
#    few_shot_qa_pairs (bool): whether to have few-shot QA pairs in the meta
#      prompt.
#    include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
#    data (list or pd.DataFrame): the raw data.
#    few_shot_index_list (list): the list of indices of few-shot examples.
#    instructions_before_exemplars (bool): whether the instruction-score pairs are
#      before the exemplars from the dataset.
#    num_score_buckets (np.inf or int): the number of score buckets when we
#      convert float accuracies to integers. Default to np.inf for not
#      bucketizing.
#    dataset_name (str): the name of the current dataset. Only used when
#      generating task description when meta_prompt_type == "instructions_only".
#    task_name (str): the name of the current task. Only used when generating task
#      description when meta_prompt_type == "instructions_only".

#   Returns:
#    meta_prompt (str): the generated meta prompt.
#   """
#   assert instruction_pos in {
#       "before_Q",
#       "Q_begin",
#       "Q_end",
#       "A_begin",
#   }, (
#       "The instruction position should be either before the question, or at the"
#       " beginning of the question, at the end of the question, or at the"
#       " beginning of the answer."
#   )
#   assert meta_prompt_type in {
#       "both_instructions_and_exemplars",
#       "instructions_only",
#   }
#   assert dataset_name in {
#       "mmlu",
#       "bbh",
#       "gsm8k",
#   }, "The lower-case dataset name must be one of mmlu, bbh, gsm8k."
#   assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)

#   meta_prompt = ""
#   if meta_prompt_type == "both_instructions_and_exemplars":
#     if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#       if instruction_pos == "A_begin":
#         meta_prompt_old_instruction_part = (
#             "Your task is to generate the answer starting sentence <Start>."
#             " Below are some previous starting sentences with their scores."
#             " The score ranges from 0 to 100.\n"
#         )
#       else:
#         meta_prompt_old_instruction_part = (
#             "Your task is to generate the instruction <INS>."
#             " Below are some previous instructions with their scores."
#             " The score ranges from 0 to 100.\n"
#         )
#     else:
#       assert optimizer_llm_name.lower() == "text-bison"
#       meta_prompt_old_instruction_part = (
#           "I have some texts along with their corresponding scores."
#           " The texts are arranged in ascending order based on their scores,"
#           " where higher scores indicate better quality.\n\n"
#       )
#     # add old instructions
#     old_instructions_and_scores_str = gen_ins_and_score_pairs_substr(
#         old_instructions_and_scores=old_instructions_and_scores,
#         old_instruction_score_threshold=old_instruction_score_threshold,
#         max_num_instructions=max_num_instructions,
#         return_str_only=True,
#         num_score_buckets=num_score_buckets,
#     )
#     meta_prompt_old_instruction_part += old_instructions_and_scores_str
#     # add QA pairs if few_shot_qa_pairs == True
#     meta_prompt_exemplar_part = ""
#     if few_shot_qa_pairs:
#       if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#         meta_prompt_exemplar_part += "Below are some problems.\n"
#       else:
#         assert optimizer_llm_name.lower() == "text-bison"
#         meta_prompt_exemplar_part += (
#             "The following exemplars show how to apply your text: you replace"
#             " <INS> in each input with your text, then read the input and give"
#             " an output. We say your output is wrong if your output is"
#             " different from the given output, and we say your output is"
#             " correct if they are the same. When replacing <INS> with an old"
#             " piece of text above, we get wrong outputs on the following"
#             " inputs.\n\n"
#         )
#       for idx in few_shot_index_list:
#         if dataset_name == "mmlu":
#           question = eval_utils._format_mmlu_example(data, idx)  # pylint: disable=protected-access
#           true_answer = data.iloc[idx, -1]
#         elif dataset_name == "bbh":
#           question = data[idx]["input"]
#           true_answer = data[idx]["target"]
#         else:
#           assert dataset_name == "gsm8k"
#           question = data.iloc[idx, 0]
#           true_answer = data.iloc[idx, 1]

#         if include_qa:  # when "Q:" and "A:" are present in the prompt
#           if instruction_pos == "before_Q":
#             meta_prompt_exemplar_part += f"\ninput:\n<INS>\nQ: {question}\nA:"
#           elif instruction_pos == "Q_begin":
#             meta_prompt_exemplar_part += f"\ninput:\nQ: <INS>\n{question}\nA:"
#           elif instruction_pos == "Q_end":
#             meta_prompt_exemplar_part += f"\ninput:\nQ: {question}\n<INS>\nA:"
#           else:  # instruction_pos == "A_begin"
#             if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#               meta_prompt_exemplar_part += f"\nQ: {question}\nA: <Start>"
#             else:
#               assert optimizer_llm_name.lower() == "text-bison"
#               meta_prompt_exemplar_part += f"\ninput:\nQ: {question}\nA: <INS>"
#         else:  # when there're no "Q:" and "A:" in the prompt
#           assert instruction_pos in {"Q_begin", "Q_end"}
#           if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#             if instruction_pos == "Q_begin":
#               meta_prompt_exemplar_part += f"\nProblem:\n<INS>\n{question}\n"
#             elif instruction_pos == "Q_end":
#               meta_prompt_exemplar_part += f"\nProblem:\n{question}\n<INS>\n"
#           else:
#             assert optimizer_llm_name.lower() == "text-bison"
#             if instruction_pos == "Q_begin":
#               meta_prompt_exemplar_part += f"\ninput:\n<INS>\n{question}\n"
#             elif instruction_pos == "Q_end":
#               meta_prompt_exemplar_part += f"\ninput:\n{question}\n<INS>\n"

#         if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#           meta_prompt_exemplar_part += (
#               f"\nGround truth answer:\n{true_answer}\n"
#           )
#         else:
#           assert optimizer_llm_name.lower() == "text-bison"
#           meta_prompt_exemplar_part += f"\noutput:\n{true_answer}\n"

#     if few_shot_qa_pairs:
#       if instructions_before_exemplars:
#         meta_prompt += (
#             meta_prompt_old_instruction_part
#             + "\n\n"
#             + meta_prompt_exemplar_part
#         )
#       else:
#         meta_prompt += (
#             meta_prompt_exemplar_part
#             + "\n\n"
#             + meta_prompt_old_instruction_part
#         )
#     else:
#       meta_prompt += meta_prompt_old_instruction_part

#     if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#       if instruction_pos == "A_begin":
#         meta_prompt += (
#             "\n\nGenerate a starting sentence that is different from all the"
#             " <Start> sentences above, and has a higher score than all the"
#             " <Start> sentences above. The starting sentence should begin with"
#             " <Start> and end with </Start>. The starting sentence should be"
#             " concise, effective, and generally applicable to all QA pairs"
#             " above."
#         )
#       else:
#         meta_prompt += (
#             "\n\nGenerate an instruction that"
#             " is different from all the instructions <INS> above,"
#             " and has a higher score than all the instructions <INS> above."
#             " The instruction should begin with <INS> and end with </INS>."
#             " The instruction should be concise, effective,"
#             " and generally applicable to all problems above."
#         )
#     else:
#       assert optimizer_llm_name.lower() == "text-bison"
#       meta_prompt += (
#           "\n\nWrite your new text that is different from the old ones and"
#           " has a score as high as possible. Write the text in square brackets."
#       )
#   else:
#     # when using a pre-trained model as optimizer
#     assert meta_prompt_type == "instructions_only"

#     assert instruction_pos in {"Q_begin", "Q_end", "A_begin"}
#     if instruction_pos == "Q_begin":
#       instruction_pos_description = "at the beginning of the question"
#     elif instruction_pos == "Q_end":
#       instruction_pos_description = "at the end of the question"
#     else:
#       assert instruction_pos == "A_begin"
#       instruction_pos_description = "at the beginning of the answer"

#     if dataset_name == "gsm8k":
#       instruction_task_description = "grade school math"
#     elif dataset_name == "mmlu":
#       instruction_task_description = task_name
#     else:
#       assert dataset_name == "bbh"
#       instruction_task_description = " ".join(task_name.split("_"))

#     meta_instruction = (
#         f"Create a piece of text {instruction_pos_description.strip()} to"
#         " enhance the precision in solving diverse"
#         f" {instruction_task_description.strip()} problems."
#     )
#     old_instructions_and_scores = sorted(
#         old_instructions_and_scores, key=lambda x: x[1]
#     )
#     old_instructions_and_scores_str = ""
#     for instruction, score, _ in old_instructions_and_scores:
#       if num_score_buckets == np.inf:
#         score_to_show = round(score, 2)
#       else:
#         score_to_show = _bucketize_float(score, num_score_buckets)
#       old_instructions_and_scores_str += (
#           f"\n\nPrecision: {score_to_show} <TEXT>{instruction}</TEXT>"
#       )
#     meta_prompt += meta_instruction + old_instructions_and_scores_str
#   return meta_prompt


# def run_evolution(**kwargs):
#   """The function for evolution."""
#   # ================= experiment configurations =============================
#   num_search_steps = kwargs["num_search_steps"]
#   old_instruction_score_threshold = kwargs["old_instruction_score_threshold"]
#   scorer_llm_dict = kwargs["scorer_llm_dict"]
#   optimizer_llm_dict = kwargs["optimizer_llm_dict"]
#   extract_final_answer_by_prompting_again = kwargs[
#       "extract_final_answer_by_prompting_again"
#   ]
#   include_qa = kwargs["include_qa"]
#   evaluate_in_parallel = kwargs["evaluate_in_parallel"]
#   tasks_all = kwargs["tasks_all"]
#   train_ratio = kwargs["train_ratio"]
#   eval_ratio = kwargs["eval_ratio"]
#   test_ratio = kwargs["test_ratio"]
#   train_index = kwargs["train_index"]
#   eval_index = kwargs["eval_index"]
#   dataset_name = kwargs["dataset_name"]
#   task_name = kwargs["task_name"]
#   num_examples = kwargs["num_examples"]
#   root_data_folder_path = kwargs["root_data_folder_path"]
#   optimizer_llm_temperature = kwargs["optimizer_llm_temperature"]
#   optimizer_llm_temperature_schedule = (
#       kwargs["optimizer_llm_temperature_schedule"]
#       if "optimizer_llm_temperature_schedule" in kwargs
#       else "constant"
#   )
#   optimizer_llm_temperature_end = (
#       kwargs["optimizer_llm_temperature_end"]
#       if "optimizer_llm_temperature_end" in kwargs
#       else None
#   )
#   initial_instructions = kwargs["initial_instructions"]
#   multiple_choice_tasks = kwargs["multiple_choice_tasks"]
#   raw_data = kwargs["raw_data"]
#   call_scorer_server_func = kwargs["call_scorer_server_func"]
#   call_optimizer_server_func = kwargs["call_optimizer_server_func"]
#   instruction_pos = kwargs["instruction_pos"]
#   prediction_treat_as_number = kwargs["prediction_treat_as_number"]
#   prediction_treat_as_bool = kwargs["prediction_treat_as_bool"]
#   result_by_instruction_folder = kwargs["result_by_instruction_folder"]
#   few_shot_qa_pairs = kwargs["few_shot_qa_pairs"]
#   num_score_buckets = kwargs["num_score_buckets"]
#   max_num_instructions = kwargs["max_num_instructions"]
#   meta_prompt_type = kwargs["meta_prompt_type"]
#   meta_prompt_instructions_before_exemplars = kwargs[
#       "meta_prompt_instructions_before_exemplars"
#   ]
#   few_shot_selection_criteria = kwargs["few_shot_selection_criteria"]
#   optimizer_llm_name = kwargs["optimizer_llm_name"]
#   num_generated_instructions_in_each_step = kwargs[
#       "num_generated_instructions_in_each_step"
#   ]
#   evaluate_generated_ins_on_few_shot = kwargs[
#       "evaluate_generated_ins_on_few_shot"
#   ]
#   num_few_shot_questions_for_instruction_refinement = kwargs[
#       "num_few_shot_questions_for_instruction_refinement"
#   ]
#   evaluate_old_ins_on_few_shot = kwargs["evaluate_old_ins_on_few_shot"]
#   eval_interval = kwargs["eval_interval"]
#   save_folder = kwargs["save_folder"]
#   verbose = kwargs["verbose"] if "verbose" in kwargs else False

#   # =================== assertions =====================
#   assert dataset_name in {
#       "mmlu",
#       "bbh",
#       "gsm8k",
#   }, "The lower-case dataset name must be one of mmlu, bbh, gsm8k."
#   assert optimizer_llm_temperature_schedule in {
#       "constant",
#       "linear_increase",
#   }, "The temperature schedule should be constant or linear_increase."

#   # =================== save configurations to json file ====================
#   configs_dict = dict()
#   configs_dict["scorer_llm_dict"] = scorer_llm_dict
#   configs_dict["optimizer_llm_dict"] = optimizer_llm_dict
#   configs_dict["instruction_pos"] = instruction_pos
#   configs_dict["optimizer_llm_temperature"] = optimizer_llm_temperature
#   configs_dict["optimizer_llm_temperature_schedule"] = (
#       optimizer_llm_temperature_schedule
#   )
#   configs_dict["optimizer_llm_temperature_end"] = optimizer_llm_temperature_end
#   with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
#     json.dump(configs_dict, f, indent=4)

#   num_servers = scorer_llm_dict["num_servers"]
#   batch_size = scorer_llm_dict["batch_size"]
#   generated_ins_on_few_shot_results_dict = dict()
#   old_ins_on_few_shot_results_dict = dict()
#   # evaluation results every a few steps
#   # format: [(i_step, instruction, detailed_results_df)]
#   eval_results = []
#   # all generated instructions, format: [(instruction, score, step_index)]
#   # the instructions that were skipped have score NaN
#   old_instructions_and_scores_raw = []
#   # the new instructions, format: [(instruction, score, step_index)]
#   old_instructions_and_scores = []
#   meta_prompts = []  # format: [(meta_prompt, step_index)]
#   instruction_score_dict = dict()  # the dictionary of {instruction: score}
#   # the dictionary of the few-shot QA indices in meta-prompt
#   # key: step index; value: the list of few-shot indices in that step
#   few_shot_index_list_by_step_dict = dict()
#   detailed_results_df_by_instruction_dict = dict()
#   wrong_questions_from_start_counter = collections.Counter()
#   # EVAL results
#   eval_detailed_results_df_dict = dict()  # {instruction: detailed_results_df}
#   instruction_eval_score_dict = dict()  # {instruction: eval_score}
#   old_instruction_md5_hashstrings_set = set()

#   print(f"tasks_all: {tasks_all}")
#   print(
#       f"train_ratio: {train_ratio}, number of training points:"
#       f" {int(num_examples * train_ratio)}"
#   )
#   print(
#       f"eval_ratio: {eval_ratio}, number of eval points: "
#       f"{int(num_examples * eval_ratio)}"
#   )
#   print(
#       f"test_ratio: {test_ratio}, number of test points: "
#       f"{int(num_examples * test_ratio)}"
#   )
#   print(
#       f"optimizer llm temperature: {optimizer_llm_temperature}, schedule:"
#       f" {optimizer_llm_temperature_schedule}"
#   )
#   print(
#       f"generating {num_generated_instructions_in_each_step} instructions in"
#       f" each step, run for {num_search_steps} steps"
#   )
#   print(
#       "discarding generated instructions with score less than:"
#       f" {old_instruction_score_threshold} (old_instruction_score_threshold)"
#   )
#   print(f"num_score_buckets: {num_score_buckets}")

#   if dataset_name == "mmlu":
#     is_multiple_choice = True
#     is_multiple_choice_eval = True
#   elif dataset_name in {"gsm8k"}:
#     is_multiple_choice = False
#     is_multiple_choice_eval = False
#   else:
#     assert dataset_name == "bbh"
#     is_multiple_choice = []
#     is_multiple_choice_eval = []
#     train_index_by_task_dict = dict()
#     eval_index_by_task_dict = dict()
#     start_index = 0
#     for task_name in tasks_all:
#       single_task_list = eval_utils.load_bbh_task_data(
#           task_name, base_dir=root_data_folder_path
#       )
#       end_index = start_index + len(single_task_list)
#       train_index_by_task_dict[task_name] = (
#           train_index[(train_index >= start_index) & (train_index < end_index)]
#           # if " - start_index" is added here, then the dict would contain
#           # indices in the original task
#       )
#       eval_index_by_task_dict[task_name] = (
#           eval_index[(eval_index >= start_index) & (eval_index < end_index)]
#           # if " - start_index" is added here, then the dict would contain
#           # indices in the original task
#       )
#       start_index = end_index
#       is_multiple_choice_single_task_train = [
#           task_name in multiple_choice_tasks
#       ] * len(train_index_by_task_dict[task_name])
#       is_multiple_choice_single_task_eval = [
#           task_name in multiple_choice_tasks
#       ] * len(eval_index_by_task_dict[task_name])
#       is_multiple_choice += is_multiple_choice_single_task_train
#       is_multiple_choice_eval += is_multiple_choice_single_task_eval

#   prev_saved_instructions = set()

#   # evaluate initial instructions
#   print("\n============== evaluating initial instructions ===============")
#   train_index = train_index[:3]
#   for instruction in initial_instructions:
#     print(f"""computing the score of "{instruction}" by prompting""")

#     detailed_results_df = eval_utils.evaluate_single_instruction(
#         data=raw_data,
#         instruction=instruction,
#         eval_index_all=train_index,
#         batch_size=batch_size,
#         call_server_func=call_scorer_server_func,
#         dataset_name=dataset_name,
#         num_servers=num_servers,
#         extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
#         include_qa=include_qa,
#         evaluate_in_parallel=evaluate_in_parallel,
#         instruction_pos=instruction_pos,
#         is_multiple_choice=is_multiple_choice,
#         prediction_treat_as_number=prediction_treat_as_number,
#         prediction_treat_as_bool=prediction_treat_as_bool,
#         prediction_num_decimals=0,
#         max_retry=120,
#         sleep_time=60,
#         verbose=verbose,
#     )

#     detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
#     scores = detailed_results_df["accuracy"]
#     average_score = np.average(scores)
#     print(f"instruction: {instruction}, score: {average_score}")
#     filename = eval_utils.instruction_to_filename(instruction)
#     file_path = os.path.join(result_by_instruction_folder, f"{filename}.csv")
#     detailed_results_df.to_csv(file_path, index=True, header=True)
#     print(f"""saving results of "{instruction}" to {file_path}""")
#     old_instructions_and_scores.append((instruction, average_score, -1))
#     old_instructions_and_scores_raw.append((instruction, average_score, -1))
#     instruction_score_dict[instruction] = average_score

#     # increment the counter on wrong questions
#     wrong_question_indices_set = set(
#         list(
#             detailed_results_df.iloc[
#                 np.where(detailed_results_df.accuracy == 0.0)[0], :
#             ].index
#         )
#     )
#     for idx in wrong_question_indices_set:
#       wrong_questions_from_start_counter[idx] += 1

#   # evolution
#   for i_step in range(num_search_steps):
#     print(f"\n================== Step {i_step} =====================")
#     if not i_step % 10:
#       print(f"old_instructions_and_scores: {old_instructions_and_scores}")

#     if optimizer_llm_temperature_schedule == "linear_increase":
#       optimizer_llm_temperature_curr = (
#           optimizer_llm_temperature
#           + i_step
#           / num_search_steps
#           * (optimizer_llm_temperature_end - optimizer_llm_temperature)
#       )
#     else:
#       optimizer_llm_temperature_curr = optimizer_llm_temperature
#     print(
#         f"current optimizer_llm_temperature: {optimizer_llm_temperature_curr}"
#     )

#     # generate new instructions
#     if few_shot_qa_pairs:
#       if few_shot_selection_criteria == "accumulative_most_frequent":
#         # select QA pairs that were done wrong the most number of times
#         most_frequent_wrong_question_indices = [
#             k
#             for k, _ in sorted(
#                 wrong_questions_from_start_counter.items(), key=lambda x: -x[1]
#             )
#         ]
#         print(
#             "len(most_frequent_wrong_question_indices):"
#             f" {len(most_frequent_wrong_question_indices)}"
#         )
#         if (
#             len(most_frequent_wrong_question_indices)
#             <= num_few_shot_questions_for_instruction_refinement
#         ):
#           few_shot_index_list = most_frequent_wrong_question_indices.copy()
#         else:
#           np.random.seed(i_step)
#           few_shot_index_list = np.sort(
#               np.random.choice(
#                   most_frequent_wrong_question_indices,
#                   num_few_shot_questions_for_instruction_refinement,
#                   replace=False,
#               )
#           )

#       elif few_shot_selection_criteria == "current_most_frequent":
#         # show exemplars done wrong most often by currently shown instructions
#         old_instruction_score_threshold_single_step = (
#             old_instruction_score_threshold if i_step > 0 else 0
#         )
#         _, old_instructions_and_scores_in_meta_prompt = (
#             gen_ins_and_score_pairs_substr(
#                 old_instructions_and_scores=old_instructions_and_scores,
#                 old_instruction_score_threshold=old_instruction_score_threshold_single_step,
#                 max_num_instructions=max_num_instructions,
#                 return_str_only=False,
#                 num_score_buckets=num_score_buckets,
#             )
#         )
#         wrong_questions_counter_single_step = collections.Counter()
#         for ins, _, _ in old_instructions_and_scores_in_meta_prompt:
#           filename = eval_utils.instruction_to_filename(ins)
#           file_path = os.path.join(
#               result_by_instruction_folder, f"{filename}.csv"
#           )
#           single_ins_df = pd.read_csv(file_path, index_col=0, header=0)
#           wrong_question_indices_set_single_old_ins = set(
#               list(
#                   single_ins_df.iloc[
#                       np.where(single_ins_df.accuracy == 0.0)[0], :
#                   ].index
#               )
#           )
#           for idx in wrong_question_indices_set_single_old_ins:
#             wrong_questions_counter_single_step[idx] += 1
#         most_occurred_wrong_questions = [
#             k
#             for k, v in wrong_questions_counter_single_step.items()
#             if v == max(wrong_questions_counter_single_step.values())
#         ]
#         if (
#             len(most_occurred_wrong_questions)
#             < num_few_shot_questions_for_instruction_refinement
#         ):
#           # pylint: disable=cell-var-from-loop
#           idx_most_to_least = sorted(
#               wrong_questions_counter_single_step,
#               key=lambda x: -wrong_questions_counter_single_step[x],
#           )
#           few_shot_index_list = idx_most_to_least[
#               :num_few_shot_questions_for_instruction_refinement
#           ]
#         else:
#           few_shot_index_list = np.sort(
#               np.random.choice(
#                   most_occurred_wrong_questions,
#                   num_few_shot_questions_for_instruction_refinement,
#                   replace=False,
#               )
#           )
#       elif few_shot_selection_criteria == "constant":
#         np.random.seed(0)
#         few_shot_index_list = np.sort(
#             np.random.choice(
#                 train_index,
#                 num_few_shot_questions_for_instruction_refinement,
#                 replace=False,
#             )
#         )
#       else:
#         assert few_shot_selection_criteria == "random"
#         np.random.seed(i_step)
#         few_shot_index_list = np.sort(
#             np.random.choice(
#                 train_index,
#                 num_few_shot_questions_for_instruction_refinement,
#                 replace=False,
#             )
#         ).tolist()

#       few_shot_index_list_by_step_dict[i_step] = few_shot_index_list

#       meta_prompt = gen_meta_prompt(
#           old_instructions_and_scores=old_instructions_and_scores,
#           instruction_pos=instruction_pos,
#           optimizer_llm_name=optimizer_llm_name,
#           old_instruction_score_threshold=old_instruction_score_threshold,
#           max_num_instructions=max_num_instructions,
#           meta_prompt_type=meta_prompt_type,
#           few_shot_qa_pairs=few_shot_qa_pairs,
#           include_qa=include_qa,
#           data=raw_data,
#           few_shot_index_list=few_shot_index_list,
#           instructions_before_exemplars=meta_prompt_instructions_before_exemplars,
#           num_score_buckets=num_score_buckets,
#           dataset_name=dataset_name,
#           task_name=task_name,
#       )

#     else:  # no few-shot exemplars in meta-prompt
#       few_shot_index_list = []
#       meta_prompt = gen_meta_prompt(
#           old_instructions_and_scores=old_instructions_and_scores,
#           instruction_pos=instruction_pos,
#           optimizer_llm_name=optimizer_llm_name,
#           old_instruction_score_threshold=old_instruction_score_threshold,
#           max_num_instructions=max_num_instructions,
#           meta_prompt_type=meta_prompt_type,
#           few_shot_qa_pairs=False,
#           include_qa=include_qa,
#           instructions_before_exemplars=meta_prompt_instructions_before_exemplars,
#           num_score_buckets=num_score_buckets,
#           dataset_name=dataset_name,
#           task_name=task_name,
#       )
#     print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
#     meta_prompts.append((meta_prompt, i_step))
#     remaining_num_instructions_to_generate = (
#         num_generated_instructions_in_each_step
#     )
#     generated_instructions_raw = []
#     while remaining_num_instructions_to_generate > 0:
#       optimizer_llm_input_text = meta_prompt
#       # generate instructions
#       print(f"current temperature: {optimizer_llm_temperature_curr}")
#       raw_outputs = call_optimizer_server_func(
#           optimizer_llm_input_text,
#           temperature=optimizer_llm_temperature_curr,
#       )

#       # Extract the generated instructions from the optimizer LLM output. Only
#       # keep some samples if the desired number of remaining instructions
#       # is smaller than the total number of decodes in this step.
#       if meta_prompt_type == "both_instructions_and_exemplars":
#         raw_outputs = raw_outputs[:remaining_num_instructions_to_generate]
#         if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"}:
#           if instruction_pos == "A_begin":
#             start_string = "<Start>"
#             end_string = "</Start>"
#           else:
#             start_string = "<INS>"
#             end_string = "</INS>"
#           for raw_output in raw_outputs:
#             if start_string not in raw_output:
#               start_index = 0
#             else:
#               start_index = raw_output.index(start_string) + len(start_string)
#             if end_string not in raw_output:
#               end_index = len(raw_output)
#             else:
#               end_index = raw_output.index(end_string)
#             new_inst = raw_output[start_index:end_index].strip()
#             generated_instructions_raw.append(new_inst)
#         else:
#           assert optimizer_llm_name.lower() == "text-bison"
#           generated_instructions_raw += [
#               extract_string_in_square_brackets(string)
#               for string in raw_outputs
#           ]

#         remaining_num_instructions_to_generate -= optimizer_llm_dict[
#             "batch_size"
#         ]
#       else:
#         assert meta_prompt_type == "instructions_only"
#         max_num_instructions_to_keep_in_each_output = 1
#         for string in raw_outputs:
#           generated_instructions_raw += parse_tag_content(string)[
#               :max_num_instructions_to_keep_in_each_output
#           ]
#         remaining_num_instructions_to_generate -= (
#             optimizer_llm_dict["batch_size"]
#             * max_num_instructions_to_keep_in_each_output
#         )

#     generated_instructions_raw = list(
#         map(eval_utils.polish_sentence, generated_instructions_raw)
#     )
#     print(f"\ninitially generated instructions: {generated_instructions_raw}\n")

#     # do not evaluate old instructions again
#     generated_instructions = []  # the new instructions generated in this step
#     for ins in generated_instructions_raw:
#       ins_md5_hashstring = eval_utils.instruction_to_filename(
#           ins, md5_hashing=True
#       )
#       if ins_md5_hashstring not in old_instruction_md5_hashstrings_set:
#         generated_instructions.append(ins)
#         old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
#       else:
#         print(f"already evaluated '{ins}' previously")
#     generated_instructions = list(set(generated_instructions))

#     to_evaluate_instructions = []
#     for instruction in generated_instructions:
#       if len(instruction) > 500:
#         print(f"Step {i_step}, instruction: {instruction}, too long, skipped")
#         continue
#       if dataset_name == "gsm8k" and any(
#           char.isdigit() for char in instruction
#       ):
#         print(
#             f"Step {i_step}, instruction: {instruction}, contains numbers,"
#             " skipped"
#         )
#         continue
#       if "INS" in instruction:
#         print(
#             f"Step {i_step}, instruction: {instruction}, contains 'INS',"
#             " skipped"
#         )
#         continue
#       to_evaluate_instructions.append(instruction)
#     print(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")

#     # evaluate new instructions on the few-shot exemplars in meta-prompt
#     if few_shot_qa_pairs and evaluate_generated_ins_on_few_shot:
#       print("evaluating GENERATED instructions on few-shot exemplars")
#       single_step_eval_on_few_shot = dict()
#       for instruction in to_evaluate_instructions:
#         if instruction not in prev_saved_instructions:
#           print(
#               f"evaluating Step {i_step}, instruction: {instruction} on"
#               " few-shot exemplars"
#           )
#           detailed_results_df = eval_utils.evaluate_single_instruction(
#               data=raw_data,
#               instruction=instruction,
#               eval_index_all=few_shot_index_list,
#               batch_size=batch_size,
#               call_server_func=call_scorer_server_func,
#               dataset_name=dataset_name,
#               num_servers=num_servers,
#               extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
#               include_qa=include_qa,
#               evaluate_in_parallel=evaluate_in_parallel,
#               instruction_pos=instruction_pos,
#               is_multiple_choice=is_multiple_choice,
#               prediction_treat_as_number=prediction_treat_as_number,
#               prediction_treat_as_bool=prediction_treat_as_bool,
#               prediction_num_decimals=0,
#               max_retry=5,
#               sleep_time=180,
#               verbose=verbose,
#           )
#           single_step_eval_on_few_shot[instruction] = detailed_results_df

#       print(
#           f"Step {i_step}, single_step_eval_on_few_shot:"
#           f" {single_step_eval_on_few_shot}\n"
#       )
#       generated_ins_on_few_shot_results_dict[i_step] = (
#           single_step_eval_on_few_shot
#       )

#     # evaluate OLD instructions on the few-shot exemplars in meta-prompt
#     if few_shot_qa_pairs and evaluate_old_ins_on_few_shot:
#       print("evaluating OLD instructions on few-shot exemplars")
#       single_step_eval_on_few_shot = dict()
#       for instruction, _, _ in old_instructions_and_scores:
#         print(
#             f"evaluating Step {i_step}, instruction: {instruction} on few-shot"
#             " exemplars"
#         )
#         detailed_results_df = eval_utils.evaluate_single_instruction(
#             data=raw_data,
#             instruction=instruction,
#             eval_index_all=few_shot_index_list,
#             batch_size=scorer_llm_dict["batch_size"],
#             call_server_func=call_scorer_server_func,
#             dataset_name=dataset_name,
#             num_servers=scorer_llm_dict["num_servers"],
#             extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
#             include_qa=include_qa,
#             evaluate_in_parallel=evaluate_in_parallel,
#             instruction_pos=instruction_pos,
#             is_multiple_choice=is_multiple_choice,
#             prediction_treat_as_number=prediction_treat_as_number,
#             prediction_treat_as_bool=prediction_treat_as_bool,
#             prediction_num_decimals=0,
#             max_retry=5,
#             sleep_time=180,
#             verbose=verbose,
#         )
#         single_step_eval_on_few_shot[instruction] = detailed_results_df

#       print(
#           f"Step {i_step}, single_step_eval_on_few_shot:"
#           f" {single_step_eval_on_few_shot}\n"
#       )
#       old_ins_on_few_shot_results_dict[i_step] = single_step_eval_on_few_shot

#     # evaluate newly generated instructions on the training set
#     for instruction in to_evaluate_instructions:
#       if instruction not in prev_saved_instructions:
#         print(f"""computing the score of "{instruction}" by prompting""")
#         detailed_results_df = eval_utils.evaluate_single_instruction(
#             data=raw_data,
#             instruction=instruction,
#             eval_index_all=train_index,
#             batch_size=batch_size,
#             call_server_func=call_scorer_server_func,
#             dataset_name=dataset_name,
#             num_servers=num_servers,
#             extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
#             include_qa=include_qa,
#             evaluate_in_parallel=evaluate_in_parallel,
#             instruction_pos=instruction_pos,
#             is_multiple_choice=is_multiple_choice,
#             prediction_treat_as_number=prediction_treat_as_number,
#             prediction_treat_as_bool=prediction_treat_as_bool,
#             prediction_num_decimals=0,
#             max_retry=5,
#             sleep_time=180,
#             verbose=verbose,
#         )
#         prev_saved_instructions.add(instruction)
#       else:
#         # do not re-evaluate instructions that had been evaluated previously
#         detailed_results_df = pd.read_csv(
#             os.path.join(result_by_instruction_folder, f"{instruction}.csv"),
#             index_col=0,
#             header=0,
#         )
#         print(f"""reading previously saved "{instruction}" information""")

#       scores = detailed_results_df["accuracy"]
#       average_score = np.average(scores)
#       print(
#           f"Step {i_step}, instruction: {instruction}, score: {average_score}"
#       )

#       # increment the counter on wrong questions
#       wrong_question_indices_set = set(
#           list(
#               detailed_results_df[detailed_results_df["accuracy"] == 0.0].index
#           )
#       )
#       for idx in wrong_question_indices_set:
#         wrong_questions_from_start_counter[idx] += 1

#       filename = eval_utils.instruction_to_filename(instruction)
#       file_path = os.path.join(
#           result_by_instruction_folder, f"""{filename}.csv"""
#       )
#       detailed_results_df.to_csv(file_path, index=True, header=True)
#       print(f"saving results to {file_path}")

#       detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
#       old_instructions_and_scores.append((instruction, average_score, i_step))
#       instruction_score_dict[instruction] = average_score

#     # record all generated instructions
#     for instruction in generated_instructions_raw:
#       if instruction in instruction_score_dict:
#         average_score = instruction_score_dict[instruction]
#       else:
#         average_score = np.nan
#       old_instructions_and_scores_raw.append(
#           (instruction, average_score, i_step)
#       )

#     # =============================== eval ====================================
#     # every eval_interval steps, evaluate the instructions that were generated
#     # in the current step and were not skipped
#     if not i_step % eval_interval:
#       for instruction in generated_instructions_raw:
#         # if the instruction wasn't skipped in any step
#         if instruction in instruction_score_dict:
#           if instruction not in instruction_eval_score_dict:
#             detailed_results_df = eval_utils.evaluate_single_instruction(
#                 data=raw_data,
#                 instruction=instruction,
#                 eval_index_all=eval_index,
#                 batch_size=batch_size,
#                 call_server_func=call_scorer_server_func,
#                 dataset_name=dataset_name,
#                 num_servers=num_servers,
#                 extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
#                 include_qa=include_qa,
#                 evaluate_in_parallel=evaluate_in_parallel,
#                 instruction_pos=instruction_pos,
#                 is_multiple_choice=is_multiple_choice_eval,
#                 prediction_treat_as_number=prediction_treat_as_number,
#                 prediction_treat_as_bool=prediction_treat_as_bool,
#                 prediction_num_decimals=0,
#                 max_retry=5,
#                 sleep_time=180,
#                 verbose=verbose,
#             )
#             eval_score = np.average(detailed_results_df["accuracy"])
#             eval_detailed_results_df_dict[instruction] = detailed_results_df
#             instruction_eval_score_dict[instruction] = eval_score
#           else:
#             eval_score = instruction_eval_score_dict[instruction]
#           print(
#               f"EVAL: \nStep {i_step}, instruction: {instruction}, eval score:"
#               f" {eval_score:.2f}"
#           )
#           eval_results.append((i_step, instruction, eval_score))

#     # ===================== save up-to-date results ===========================
#     results_dict = dict()
#     results_dict["meta_prompts"] = meta_prompts
#     results_dict["old_instructions_and_scores"] = list(
#         old_instructions_and_scores
#     )
#     results_dict["old_instructions_and_scores_raw"] = list(
#         old_instructions_and_scores_raw
#     )
#     results_dict["generated_ins_on_few_shot_results_dict"] = (
#         generated_ins_on_few_shot_results_dict
#     )
#     results_dict["old_ins_on_few_shot_results_dict"] = (
#         old_ins_on_few_shot_results_dict
#     )
#     results_dict["few_shot_index_list_by_step_dict"] = (
#         few_shot_index_list_by_step_dict
#     )
#     results_dict["eval_results"] = eval_results
#     results_dict["eval_detailed_results_df_dict"] = (
#         eval_detailed_results_df_dict
#     )
#     with open(os.path.join(save_folder, "results_dict.pkl"), "wb") as fp:
#       pickle.dump(results_dict, fp)
#     print(f"\nsaved all results to\n{save_folder}")


META_INSTRUCTION = [
  [
    "I have some texts along with their corresponding scores."
    " The texts are arranged in ascending order based on their scores,"
    " where higher scores indicate better quality.\n\n", 

    "\n\nWrite your new text that is different from the old ones and"
    " has a score as high as possible. Avoid overly long texts. "
    "Write the text in square brackets."
  ],
  [
    "I have some texts, each accompanied by two scores: a relevance score and an aesthetics score."
    " Higher scores indicate better quality for their respective criteria. \n\n", 
    
   "\n\nWrite your new text that is different from the old ones to"
    " maximize both the relevance score and the aesthetics score simultaneously. "
    "Avoid overly long texts. "
    "Write the text in square brackets."
  ],
]
  

def gen_meta_prompt_T2I(
    old_instructions_and_scores,
    aggregate_scores=True,
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
  old_instructions_and_scores_str = gen_ins_and_score_pairs_substr(
      old_instructions_and_scores=old_instructions_and_scores,
      aggregate_scores=aggregate_scores,
      old_instruction_score_threshold=old_instruction_score_threshold,
      max_num_instructions=max_num_instructions,
      return_str_only=True,
      # num_score_buckets=num_score_buckets,
  )
  #TODO: redesign the meta instructions
  meta_ins1, meta_ins2 = META_INSTRUCTION[0] if aggregate_scores else META_INSTRUCTION[1]

  meta_prompt = meta_ins1 + old_instructions_and_scores_str + meta_ins2

  return meta_prompt


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
    buffer = BytesIO()
    img.save(buffer, format="PNG")
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
    meta_prompts = []  # format: [(meta_prompt, step_index)]
    old_instruction_md5_hashstrings_set = set() # to avoid re-evaluating instructions
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

      old_instructions_and_scores.append((p, curr_sc, -1))

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
      meta_prompt = gen_meta_prompt_T2I(
          old_instructions_and_scores=old_instructions_and_scores,
          aggregate_scores=aggregate_scores,
          old_instruction_score_threshold=old_instruction_score_threshold,
          max_num_instructions=max_num_instructions,
      )

      if not i_step % 2:
        print("\n**************************************************")
        print(f"*meta_prompt: \n\n{meta_prompt}\n")
        print("**************************************************\n", flush=True)

      meta_prompts.append((meta_prompt, i_step))
      remaining_num_instructions_to_generate = num_generated_instructions_in_each_step
      generated_instructions_raw = []
      # client = openai.OpenAI(api_key="<your_openai_api_key>", base_url="<your_openai_api_base>")
      tik = time.time()
      while remaining_num_instructions_to_generate > 0:
        optimizer_llm_input_text = meta_prompt
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
        if len(instruction) > 1000:
          print(f"Step {i_step}, {iii}-th instruction, too long, skipped")
          continue
        if instruction.startswith("Text: " or "text: "):
          instruction = instruction[len("Text: "):]

        to_evaluate_instructions.append(instruction)
      print(f"\nnumber of to-evaluate generated instructions: {len(to_evaluate_instructions)}\n")
      if len(to_evaluate_instructions) == 0:
        continue

      # evaluate these newly generated prompts: 
      orig_query_ls = [triplet[0] for triplet in old_instructions_and_scores if triplet[-1] == -1]
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
        print(f"Step {i_step}, avg_score: relevance: {cum_rel/len(score_ls)}, aesthetics: {cum_aes/len(score_ls)}")      

      # save this step
      for j, p in enumerate(to_evaluate_instructions):
        old_instructions_and_scores.append((p, score_ls[j], i_step))

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
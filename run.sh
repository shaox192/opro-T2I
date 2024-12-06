#!/bin/bash

which python

# python ./opro/optimization/optimize_instructions.py \
#        --optimizer="gpt-3.5-turbo" \
#        --scorer="gpt-3.5-turbo" \
#        --instruction_pos="Q_begin" \
#        --dataset="gsm8k" \
#        --task="train" \
#        --meta_prompt_type="both_instructions_and_exemplars" \
#        --palm_api_key="<your_palm_api_key>" \
#        --openai_api_key="<your_openai_api_key>"


python ./opro/optimization/optimize_instructions_T2I.py \
       --optimizer="gpt-4o-mini" \
       --scorer="relevance" \
       --dataset="diffusionDB" \
       --save-dir="." \
       --param-aggregate-scores=False \
       --param-subset-size 2 \
       --param-num-search-steps 3 \
       --param-num-gen-per-search 2 \
       --openai_api_key="sk-BOa6EI5QKURnRvb574A86b1b24Fc445dB95eDaEe82F7C9F9" \
       --openai_api_base="https://api.fantasyfinal.cn/v1"

exit


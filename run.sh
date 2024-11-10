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
       --dataset="mscoco" \
       --param-num-search-steps 3 \
       --param-num-gen-per-search 4 \
       --openai_api_key="<your_openai_api_key>"
       --openai_api_base=""<your_openai_api_base>"

exit


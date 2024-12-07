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

AGGREGATE_SCORES=True

python ./opro/optimization/optimize_instructions_T2I.py \
       --optimizer="gpt-4o-mini" \
       --scorer="relevance" \
       --dataset="diffusionDB" \
       --save-dir="." \
       --param-aggregate-scores=$AGGREGATE_SCORES \
       --param-subset-size 20 \
       --param-num-search-steps 10 \
       --param-num-gen-per-search 3 \
       --openai_api_key="<>" \
       --openai_api_base="<>" \
       > out-agg-${AGGREGATE_SCORES}.txt


exit


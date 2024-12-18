#!/bin/bash

which python

AGGREGATE_SCORES=True
MULTI_MODAL=True

python ./opro/optimization/optimize_instructions_T2I.py \
       --optimizer="gpt-4o-mini" \
       --scorer="relevance" \
       --dataset="diffusionDB" \
       --save-dir="." \
       --param-aggregate-scores=$AGGREGATE_SCORES \
       --param-multi-modal=$MULTI_MODAL \
       --param-subset-size 20 \
       --param-num-search-steps 10 \
       --param-num-gen-per-search 3 \
       --openai_api_key="" \
       # > out-agg-${AGGREGATE_SCORES}-multimodal-${MULTI_MODAL}.txt

#        --openai_api_base="<>" \

exit


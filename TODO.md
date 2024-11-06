
# CS 546 nlp proj todo list


### Todo

- [ ] VLM and Scorer
    This should idealy go into [prompt_utils.py](./opro/prompt_utils.py). I have a placeholder there that basically gives a random number. We would probably want to include relevance first as implemented as a clip loss between images and original user prompts. 

- [ ] Datasets, and the interface for the datasets. 
    I have a skeleton in [data_utils.py](./opro/data_utils.py)
    Right now it loads ground truth images too (at least for mscoco we have expected images), but this is may not be necessary.

- [ ] Some visualization for the slides
    Do we want to at least run a couple images, show the results, and maybe also an optimization curve from the scores?

### Updates

#### 11/5/2024

Use [run.sh](./run.sh) to run the T2I optimization. Need to replace the openai_key with yours.

```sh
bash run.sh
```
I only included some demo examples from [mscoco](./data/mscoco/). I downloaded 2 images, each with 5 original captions that can be used as inital prompts. You should expect a new "output" folder created. You should find a csv file with results that look like this:

| instructions  | scores | step |
| dozens of people ... | 94 | -1 |
| a crowd of people ... | 18 | 0 |
| enthusiastic crowds ... | 67 | 1 |

The first column is the prompt to be fed through a VLM, second columns is its score (e.g. relevance, aesthetics). Currently, it only allows one type of score, we can allow more in the future. The third column is the step index. -1 indicates that this prompt is one of the initial prompts (e.g. original caption of mscoco images). 0 means this prompt was generated in the very first round of meta-prompting evolution cycle. 1 means it is from the second round etc. 

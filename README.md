# Thunder-KoNUBench
This repository contains the code for the paper "Thunder-KoNUBench: A corpus-aligned benchmark for Korean Negation Understanding". Thunder-KoNUBench dataset is also publicly available at <https://huggingface.co/datasets/thunder-research-group/SNU_Thunder-KoNUBench>

## Install & SetUp
~~~bash
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ~

git clone https://github.com/mcrl/Thunder-KoNUBench.git
cd Thunder-KoNUBench
~~~

### Add Tasks to lm-eval-harness
To use Thunder-KoNUBench tasks inside lm-eval-harness, copy the task directories into your lm-eval-harness installation:
~~~bash
bash add_tasks.sh
~~~

### Evaluation
To evaluate model performance on Thunder-KoNUBench, use the following command:
~~~bash
lm_eval \
    --model hf \
    --model_args pretrained={Model_Name},trust_remote_code=True \
    --tasks ko_nubench_symbol,ko_nubench_cloze \
    --output_path {Directory_to_store_results} \
    --log_samples \
    --batch_size auto
~~~

### Error Analysis
For error analysis after evalation, use the following command:
~~~bash
# Specify the setting to analyze (baseline or sft)
python analyze.py baseline
~~~

### Supervised-Fine-Tuning
Thunder-KoNUBench also provides training data for fine-tuning models on negation understanding. Below is an example using torchrun with distributed training:
~~~bash
  torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --node_rank=$NODE_RANK \
  --master_port=8008 \
  sft/main.py \
    --model {Model_Name} \
    --global_batch 128 \
    --micro_batch 16 \
    --learning_rate 3e-5 \
    --seed 1200 \
    --deepspeed_stage 1 \
    --dtype bf16 \
    --num_cpus 16 \
    --checkpoint_interval 200 \
    --checkpoint_path {Directory_to_save_ckp} \
    --wandb_off \
    --epochs 3
~~~

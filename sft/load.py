
import os
import time as t

import torch
import torch.distributed as dist
import torchinfo
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from peft import LoraConfig, get_peft_model, TaskType

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def load_model_tokenizer(args, summary=False):
    model_loading_start = t.time()
    
    # model
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    if args.activation_recomputation:
        model.gradient_checkpointing_enable()
    
    if summary:
        torchinfo.summary(model)
    
    model_loading_end = t.time()
    model_loading_duration = round(model_loading_end - model_loading_start, 2)
    args.debug_model_loading_duration = model_loading_duration

    # tokenizer
    tok_loading_start = t.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # handle special cases
        if  "qwen" in args.model:
            # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
            tokenizer.pad_token = "<|endoftext|>"
        elif (
            tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
            or tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
        ):
            assert tokenizer.pad_token_id == 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok_loading_end = t.time()
    tok_loading_duration = round(tok_loading_end - tok_loading_start, 2)
    args.debug_tokenizer_loading_duration = tok_loading_duration    

    return model, tokenizer

def build_peft_model(base_model):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False
    )
    peft_model = get_peft_model(base_model, lora_cfg)  # freeze the base model automatically
    peft_model.print_trainable_parameters()            
    return peft_model

def tokenize_load_dataset_cloze(args, tokenizer: AutoTokenizer):
    prompt = f"문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n-말다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문:"
    train_dataset = []
    with open(file=args.dataset_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        ctx = f'{prompt} {d["original_sentence"]}\n부정문:'

        ctx_tokenized = tokenizer(ctx, add_special_tokens=True)
        ctx_ids = ctx_tokenized["input_ids"]

        end = f' {d["standard_negation"]}'
        end_tokenized = tokenizer(end, add_special_tokens=False)
        end_ids = end_tokenized["input_ids"]

        seq = ctx_ids + end_ids
        target = ([-100] * len(ctx_ids)) + end_ids


        train_dataset.append({
            "context": d["original_sentence"],
            "ending": d["standard_negation"],
            "input_ids": seq,
            "target": target
        })

    return train_dataset

def tokenize_load_dataset_symbol(args, tokenizer: AutoTokenizer):
    import random
    prompt = f"standard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n-말다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정한 문장을 고르시오.\n원문:"
    train_dataset = []
    with open(file=args.dataset_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        ctx = f'{prompt} {d["original_sentence"]}\n'

 
        std = d["standard_negation"]
        local = d["local_negation"]
        contra = d["contradiction"]
        para = d["paraphrase"]
        if (local == None) or (local == ""):
            choices_str = [std, contra, para]
            choices = ["A", "B", "C"]
        else:
            choices_str = [std, local, contra, para]
            choices = ["A", "B", "C", "D"]        

        idx = d["idx"]
        if idx.startswith("G"):
            idx = idx[1:]
        random.seed(int(idx))
        random.shuffle(choices_str)
        

        for i in range(len(choices)):
            ctx += f"{choices[i]}. {choices_str[i]}\n"
        ctx += "\n정답:"
        
        
        ctx_tokenized = tokenizer(ctx, add_special_tokens=True)
        ctx_ids = ctx_tokenized["input_ids"]

        
        gold = choices_str.index(std)
        end = f' {choices[gold]}. {d["standard_negation"]}'
        end_tokenized = tokenizer(end, add_special_tokens=False)
        end_ids = end_tokenized["input_ids"]

        seq = ctx_ids + end_ids
        target = ([-100] * len(ctx_ids)) + end_ids


        train_dataset.append({
            "context": d["original_sentence"],
            "ending": d["standard_negation"],
            "input_ids": seq,
            "target": target
        })

    return train_dataset

def deepspeed_init(args):
    deepspeed.init_distributed(verbose=False)
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.global_rank = torch.distributed.get_rank()
    args.local_rank = int(os.getenv("LOCAL_RANK", "0")) 
    
    
def deepspeed_destroy():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    
def load_deepspeed_config(args):
    loading_start = t.time()
    
    # dataset size (after distributed init)
    grad_accum = args.global_batch // args.micro_batch // args.world_size
    assert args.global_batch == grad_accum * args.micro_batch * args.world_size, f"grad_accum: {grad_accum}, global_batch: {args.global_batch}, micro_batch: {args.micro_batch}, world_size: {args.world_size}"
    
    steps_per_epoch = args.num_dataset_rows // args.global_batch
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps*args.warmup_ratio)
    
    # (save to args)
    args.grad_accum = grad_accum
    args.steps_per_epoch = steps_per_epoch
    args.total_steps = total_steps
    
    ds_config = {
        "zero_optimization":{
            "stage":args.deepspeed_stage,
            "gather_16bit_weights_on_model_save": True if args.deepspeed_stage==3 else False
        },
        "train_batch_size": args.global_batch,
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps":grad_accum, 
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0, 
            "initial_scale_power": 16,  
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
        "optimizer": {
            "type": "AdamW", 
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999], 
                    "eps":1e-8, 
                    "weight_decay": 0.01,
                    # "max_grad_norm ":1.0, # added
            }
        },
        "gradient_clipping": 1.0,
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps":total_steps,
                "warmup_num_steps": warmup_steps,
                "warmup_type":"linear", 
            }
        },
        "activation_checkpointing": {
            "partition_activations": True if args.activation_recomputation else False,
            "cpu_checkpointing": True if args.activation_recomputation else False,
            "contiguous_memory_optimization": True if args.activation_recomputation else False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
    return ds_config


def load_checkpoint(args, model_engine):
    raise NotImplementedError("load_checkpoint is not implemented yet.")

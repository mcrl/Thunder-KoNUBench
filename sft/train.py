import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import wandb
from utils import sanitize_model_name

def _pad_stack(seqs, pad_token_id: int):
    # seqs: List[LongTensor [T]]
    # print(f'pad_token_id is {pad_token_id}')
    return pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)

def _make_attn_mask(input_ids, pad_token_id: int):
    # 1 for real token, 0 for pad
    return (input_ids != pad_token_id).long()

def sft_collate_fn(batch, pad_token_id):
    # batch: [{"input_ids": [...], "target": [...]} ...]
    input_id_list = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    label_list    = [torch.tensor(b["target"],    dtype=torch.long) for b in batch]

    input_ids = _pad_stack(input_id_list, pad_token_id)              # [B, T_max]
    labels    = _pad_stack(label_list,   -100)                       # [B, T_max], pad → -100
    attn_mask = _make_attn_mask(input_ids, pad_token_id)             # [B, T_max]

    return {"input_ids": input_ids, "target": labels, "attention_mask": attn_mask}

def sft_process_batch(args, batch):
    input_ids_full =      batch["input_ids"][:, :args.model_seq_len + 1]
    target_full    =         batch["target"][:, :args.model_seq_len + 1]
    attn_mask_full = batch["attention_mask"][:, :args.model_seq_len + 1]

    input_ids = input_ids_full[:,  :-1].cuda()
    target    =    target_full[:, 1:  ].cuda()
    attn_mask = attn_mask_full[:,  :-1].cuda()

    assert input_ids.shape == target.shape == attn_mask.shape
    
    return input_ids, target, attn_mask



# dataloader
def load_dataloader(args, dataset, collate_fn, rank):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, 
        num_replicas=args.world_size, 
        rank=rank, # IMPORTANT !!!!!! AVOID HANGING
        seed=args.seed,
        shuffle=True,
        drop_last=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(args.micro_batch),
        collate_fn=collate_fn,  # raw
        shuffle=False,
        num_workers=args.num_cpus,
        persistent_workers=False, 
        sampler=sampler,
    )
    return dataloader, sampler
    

# forward for Pretraining / SFT
def forward_one_step(model_engine, input_ids, target, attn_mask):
    # forward
    logits = model_engine(
        input_ids=input_ids,
        attn_mask=attn_mask
        # token_type_ids=token_type_ids
    ).logits

    # logits = model_engine(
    #     input_ids=input_ids,
    #     attention_mask=attn_mask
    #     # token_type_ids=token_type_ids
    # ).logits
    logits = logits.contiguous().view(-1, logits.shape[-1])
    target = target.contiguous().view(-1)
    loss = torch.nn.functional.cross_entropy(logits, target)
    
    return loss


def train_one_epoch(
    args,
    tokenizer,
    model_engine, 
    dataset,
    epoch,
    initial_idx,
    global_step,
    ):
    checkpoint_path = f"{args.checkpoint_path}/{args.model}/seed{args.seed}-lr{args.learning_rate}"

    # collate fn
    collate_fn = partial(sft_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    # sliced dataset?
    if initial_idx > 0:
        # NOTE: "dataset" index
        sliced_dataset = dataset.select(range(initial_idx, len(dataset)))
        dataloader, sampler = load_dataloader(
            args=args,
            dataset=sliced_dataset,
            collate_fn=collate_fn,
            rank=args.global_rank,
        )
    else:
        dataloader, sampler = load_dataloader(
            args=args,
            dataset=dataset,
            collate_fn=collate_fn,
            rank=args.global_rank,
        )
        
    # iterator (progressbar)
    if args.global_rank == 0:
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            ncols=90,
            initial=initial_idx
        )
    else:
        pbar = enumerate(dataloader)
        
    # train
    sampler.set_epoch(epoch)
        
    max_gpu_mem = None
    cur_step = initial_idx
    for step_idx, batch in pbar:
        # batch
        input_ids, target, attn_mask = sft_process_batch(args, batch)
        
        # forward
        loss = forward_one_step(model_engine, input_ids, target, attn_mask=attn_mask) 
        # backward
        model_engine.backward(loss)
        model_engine.step()
        cur_step += 1
                    
        if cur_step % args.grad_accum == 0 and cur_step > 0:
            global_step += 1
            
            if args.global_rank == 0:
                if global_step == 2:
                    max_gpu_mem = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 2)
                    
                pbar.set_postfix(
                    loss=loss.item(),
                    max_gpu_mem=max_gpu_mem,
                    gs=global_step,     
                )
        
                curent_lr = model_engine.optimizer.param_groups[0]['lr']
                if not args.wandb_off:
                    wandb.log({
                        "global_step_custom":global_step,
                        "train_loss": loss.item(),
                        "learning_rate":curent_lr,
                    })
            
        if cur_step % args.checkpoint_interval == 0 and cur_step > 0:
            if global_step % args.checkpoint_interval == 0:
                print(f"[INFO] Checkpoint saved at step {global_step} | rank {args.global_rank}")
                print(f"global_step % args.checkpoint_interval : {global_step % args.checkpoint_interval}")
    
    return model_engine, global_step
        
        
def train(
    args,
    tokenizer,
    model_engine, 
    dataset
):  

    initial_idx = 0
    global_step = 0
    if args.prev_ckpt is not None:
        pass
    # TODO: load checkpoint
    
    for epoch in range(args.epochs):
        
        print(f"[INFO] Training started: epoch {epoch+1}/{args.epochs}")
        model_engine, global_step = train_one_epoch(
            args=args,
            tokenizer=tokenizer,
            model_engine=model_engine, 
            dataset=dataset,
            epoch=epoch,
            initial_idx=initial_idx,
            global_step=global_step,
        )

        torch.distributed.barrier()

        save_dir = f"{args.checkpoint_path}/{sanitize_model_name(args.model)}-seed{args.seed}-lr{args.learning_rate}/epoch{epoch}"
      
        if torch.distributed.get_rank() == 0:
            print(f"[RANK0] Saving checkpoint to '{save_dir}'")
            
            model_engine.module.save_pretrained(save_dir, safe_serialization=True)
            model_engine.module.config.save_pretrained(save_dir)  # config.json
            tokenizer.save_pretrained(save_dir)  
            print("[RANK0] saved files:", os.listdir(save_dir))    
        torch.distributed.barrier()
    
    return

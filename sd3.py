import os
import ast
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline


model_local_path = "/workspace/self-cross-edit/model"


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def read_prompts(prompt_path):
    all_caption_layout = {}
    with open(prompt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::::")
            if len(parts) == 2:
                caption_key = parts[0].strip()
                value_str = parts[1].strip()
                try:
                    token_indices = ast.literal_eval(value_str)
                except (ValueError, SyntaxError):
                    token_indices = value_str
                all_caption_layout[caption_key] = token_indices
    return all_caption_layout


@torch.no_grad()
def sampling(args, prompts, seed, pipe, rank, world_size, device):
    total = len(prompts)
    prompts_per_gpu = (total + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        PROMPT = prompts[idx]
        save_dir = os.path.join(args.output_path, args.sub_name, PROMPT)
        os.makedirs(save_dir, exist_ok=True)

        print(f"GPU {rank} | Processing prompt {idx - start + 1}/{end - start}: {PROMPT}")

        SEED = seed
        print(f"GPU {rank} | Seed ({SEED})")
        generator = torch.Generator(device).manual_seed(SEED)
        images = pipe(
            prompt=PROMPT,
            generator=generator,
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images
        images[0].save(os.path.join(save_dir, f"{SEED}.png"))

    print(f"GPU {rank} has completed all tasks")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="complex")
    parser.add_argument("--output_path", type=str, default="/workspace/self-cross-edit/output_sd3/image")
    #parser.add_argument("--seeds", type=int, nargs="+", default=[1887901197, 311832740])
    args = parser.parse_args()

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_local_path,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    #seeds = args.seeds
    for i in range(4):
        seed = random.randint(0, 1000000000)
        seed_everything(seed)
        file_name = args.file_name + "_val.txt"
        prompt_path = os.path.join("T2I", file_name)
        args.sub_name = prompt_path.split("/")[-1].split(".")[0]

        all_caption_layout = read_prompts(prompt_path)
        prompt_list = list(all_caption_layout.keys())

        sampling(args, prompt_list, seed, pipe, rank, world_size, device)

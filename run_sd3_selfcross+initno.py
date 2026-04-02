import ast
import random
import re
import torch
from pipelines.sd3pipline import StableDiffusion3Pipeline
import json
import os
import sys
sys.setrecursionlimit(5000)
run_sd  =  False
run_initno = False
#model_choice = "stabilityai/stable-diffusion-3-medium-diffusers"  #"stabilityai/stable-diffusion-3.5-medium", #
#run_sd3_selfcross+initno.py


def read_prompts(prompt_path):
    all_caption_layout = {}
    with open(prompt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 去除行末的换行符和首尾空格
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


import re


def parse_string_to_word_pairs(pair_str):
    """
    将没有引号的字符串 "[[black,white,cat],[white,sink]]" 解析为 Python 列表:
    [['black', 'white', 'cat'], ['white', 'sink']]
    """
    matches = re.findall(r'\[([a-zA-Z0-9_,\s]+)\]', str(pair_str))
    result = []
    for m in matches:
        words = [w.strip() for w in m.split(',')]
        if len(words) > 1:
            result.append(words)
    return result


def get_t5_token_indices(prompt, word_groups, tokenizer):
    """
    将单词列表映射为 T5 tokenizer 的 token 索引，并解决“重复单词”和“单词被拆分为多个token”的问题。
    如果单词只对应 1 个 token，返回单个数字；如果被拆分为多个 token，则返回 "a:b" 格式（左闭右开切片）。
    如果找不到对应的 Token，将其从 word_groups 中剔除。
    返回: (mapped_groups, updated_word_groups)
    """
    tokens = tokenizer.tokenize(prompt)

    # 记录已经被匹配过的 token 索引 (0-based)
    used_indices = set()
    mapped_groups = []
    updated_word_groups = []  # 用于存储剔除未匹配单词后的纯净组

    for group in word_groups:
        mapped_group = []
        updated_group = []  # 当前组更新后的单词列表

        for word in group:
            target_word = word.lower().strip()
            matched_start = None
            matched_end = None

            # 遍历 tokens，寻找可以拼接成 target_word 的 token 序列
            for i in range(len(tokens)):
                if i in used_indices:
                    continue

                current_str = ""
                # 向后尝试拼接 token
                for j in range(i, len(tokens)):
                    if j in used_indices:
                        break  # 如果遇到已经被用的 token，终止当前起点的拼接

                    clean_token = tokens[j].replace(' ', '').replace('▁', '').lower()
                    current_str += clean_token

                    if current_str == target_word:
                        matched_start = i
                        matched_end = j
                        break
                    # 如果拼接后的字符串不再是 target_word 的前缀，提前终止
                    elif not target_word.startswith(current_str):
                        break

                if matched_start is not None:
                    break

            if matched_start is not None:
                # 标记这段区间内的 token 为已使用
                for k in range(matched_start, matched_end + 1):
                    used_indices.add(k)


                if matched_start == matched_end:
                    mapped_group.append(matched_start)
                else:
                    # 如果跨越了多个 token，使用 a:b 格式追加 (这里采用 Python 惯用的左闭右开区间 a:b)
                    mapped_group.append(f"{matched_start}:{matched_end + 1}")
                # ======================

                updated_group.append(word)  # 匹配成功，保留该单词
            else:
                print(f"[警告] 找不到未被使用的对应 Token: '{word}' in '{prompt}'，已将其剔除。")

        # 只要该组里还有匹配成功的单词
        if mapped_group:
            mapped_groups.append(mapped_group)
            updated_word_groups.append(updated_group)

    print(updated_word_groups)

    # 返回两个列表
    return mapped_groups, updated_word_groups

def Convert(string):
    string = string.replace(",", " , ")
    string = string.replace(".", "")
    li = list(string.split(" "))
    return li

# pipe = StableDiffusion3Pipeline.from_pretrained(model_choice,
#     torch_dtype=torch.bfloat16,
#     token = "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"
# )

model_local_path = "/root/autodl-tmp/model/stable-diffusion-3-medium-diffusers"

# 2. 改为本地读取
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_local_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe = pipe.to("cuda:0")



prompt_path = "T2I/color_val.txt"
sub_name = prompt_path.split("/")[-1].split(".")[0]
all_caption_layout = read_prompts(prompt_path)

for i,PROMPT in enumerate(all_caption_layout.keys()):

    if i >= 0:

        path1 = f'/root/autodl-tmp/image/{sub_name}/{PROMPT}'
        path2 = f'/root/autodl-tmp/attention/{sub_name}/{PROMPT}'
        #words = Convert(PROMPT)

        raw_pair_str = all_caption_layout[PROMPT]
        word_groups = parse_string_to_word_pairs(raw_pair_str)
        token_indices, word_groups = get_t5_token_indices(PROMPT, word_groups, pipe.tokenizer_3)

        print(PROMPT, token_indices)
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)

        SEED = random.randint(0, 2**32-1)
        print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))
        generator = torch.Generator("cuda").manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        images = pipe(prompt=PROMPT, token_indices=token_indices, guidance_scale=4.5, generator=generator,
                      num_inference_steps=28, max_iter_to_alter=14,attention_res = 64, from_where=[5,6,7,8,9,10,11,12],
                      result_root=path2, K=16, seed=SEED, run_sd=run_sd, run_initno=run_initno,words=word_groups).images
        images[0].save(path1 + f"/{SEED}.png")

    else:
        print(i)



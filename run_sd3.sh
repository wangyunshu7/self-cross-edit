#!/bin/bash

# 将 "your_script.py" 替换为你实际的 Python 文件名
SCRIPT_NAME="sd3.py"

# 定义对应的 file_name 数组
FILES=("o" "ao")

echo "开始在不同的显卡上并行启动生成任务..."

# 使用 for 循环遍历启动
for i in "${!FILES[@]}"; do
    file=${FILES[$i]}
    # 为每个进程分配不同的通讯端口，防止本地端口冲突
    port=$((29500 + i))
    
    echo " -> 正在 GPU ${i} 上启动 file_name: ${file} (master_port: ${port})"
    
    # 核心命令：
    # 1. 用 CUDA_VISIBLE_DEVICES 限制该进程只能看到一张卡
    # 2. 用 torchrun --nproc_per_node=1 伪装成单机单卡的分布式运行以满足代码中的 nccl 需求
    CUDA_VISIBLE_DEVICES=$i torchrun \
        --nproc_per_node=1 \
        --master_port=$port \
        $SCRIPT_NAME --file_name "$file" &
done

# 等待所有后台任务运行完成
wait

echo "所有生成任务已执行完毕！"
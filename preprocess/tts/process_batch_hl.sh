#!/bin/bash
#SBATCH --partition=IAI_SLURM_HGX
#SBATCH --job-name=preproc
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --cpus-per-task=64
#SBATCH --time 72:00:00

# 批次数量
num_batches=4


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python 脚本路径
python_script="./process_text_hl.py"

# 循环启动 10 个 Python 进程，每个进程处理一个批次
for i in $(seq 0 $((num_batches-1))); do
    # 启动一个后台 Python 进程来处理每个批次
   CUDA_VISIBLE_DEVICES=${GPULIST[$i]}  python3 $python_script --batch_id $i & 
done

# 等待所有后台进程完成
wait

echo "所有批次处理完成！"


#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --partition=HGX,DGX
##SBATCH --exclude=hgx-hyperplane[02]
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=tts1.out
#SBATCH --error=tts1.error.out

# 批次数量
num_batches=10


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python 脚本路径
python_script="./process_text1.py"

# 循环启动 10 个 Python 进程，每个进程处理一个批次
for i in $(seq 0 $((num_batches-1))); do
    # 启动一个后台 Python 进程来处理每个批次
   CUDA_VISIBLE_DEVICES=0  python3 $python_script --batch_id $i --num_batch $num_batches & 
done

# 等待所有后台进程完成
wait

echo "所有批次处理完成！"


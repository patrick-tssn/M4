#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --partition=HGX,DGX
##SBATCH --exclude=hgx-hyperplane[02]
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=tts1ori5.out
#SBATCH --error=tts1ori5.error.out

# 批次数量
num_batches=5


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python 脚本路径
python_script="./process_text1.py"

CUDA_VISIBLE_DEVICES=0  python3 $python_script --batch_id 4 --num_batch $num_batches


# # 等待所有后台进程完成
# wait

# echo "所有批次处理完成！"


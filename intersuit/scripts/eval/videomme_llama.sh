#!/bin/bash
#SBATCH --job-name=e-LVsub
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/videomme-longva-llama3sub.out
#SBATCH --error=./slurm_logs/videomme-longva-llama3sub.error.out

# export PYTHONPATH=$(pwd)
# echo $PYTHONPATH

# LLM_VERSION="checkpoints/Qwen2-7B-Instruct-224K"
# LLM_VERSION="checkpoints/Qwen2-7B-Instruct"
LLM_VERSION="checkpoints/Meta-Llama-3.1-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune_llavanext"
# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune_llavanext_freezeclip"
# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune_llavanext_unfreezeclip"
# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-dpo_llavanext"
# MID_RUN_NAME="longva7b-llavanext-llama31"
MID_RUN_NAME="longva7b-llavanextsub10k-llama31-rev"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


model_path="checkpoints/${MID_RUN_NAME}"
cache_dir="./cache_dir"
Zero_Shot_QA="./inputs/eval"
video_dir="${Zero_Shot_QA}/videomme/videos"
gt_file_question="${Zero_Shot_QA}/videomme/test_q.json"
gt_file_answers="${Zero_Shot_QA}/videomme/test_a.json"
output_dir="${Zero_Shot_QA}/videomme/${MID_RUN_NAME}"
NUM_FRAMES=32

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

    #   --model_base ${model_base} \

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m longva.eval.model_videoqa_mc \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_frames $NUM_FRAMES \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done
wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

python scripts/eval/eval_acc.py --src $output_file --dst $output_dir/evaluation.json

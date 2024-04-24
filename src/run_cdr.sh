
cd /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data
export MODEL_PATH="/home/zhangtaiyan/workspace/comp/used_data/GoogleNews-vectors-negative300.bin"

original_submission_path1="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/submission.jsonl"
# factory生成的预测文件地址
# factory_pred_path1="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-7B/pt/sft/full/2024-03-29-17-53-22-wo_A-1e-6/generated_predictions.jsonl"
# # factory生成的预测文件地址
# factory_pred_path2="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral7B_instruction/pt/sft/full/2024-04-12-19-02-01-wo_A-1e-6/generated_predictions.jsonl"
factory_pred_path1=/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-7B/pt/sft/full/2024-04-15-11-07-29-wo_A-1e-6/checkpoint-280/generated_predictions.jsonl
# factory生成的预测文件地址
factory_pred_path2=/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral7B_instruction/pt/sft/full/2024-04-15-11-20-05-wo_A-1e-6/checkpoint-120/generated_predictions.jsonl
# 生成的submission文件地址
submit_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/fuse_submission/golden/submission.jsonl"
candidate_file_path="/home/zhangtaiyan/workspace/comp/LinkBERT/submission.jsonl"
# submit_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/BioMistral-7B/pt/sft/full/2024-04-05-00-02-39-wo_A-1e-6/submission_rep.jsonl"
# 答案文件地址
answer_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data.jsonl"
refined_answer_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data_refined.jsonl"
# 分数报告地址
score_report_path="$(dirname "$factory_pred_path")/score.txt"
refined_score_report_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/fuse_submission/duplicate_task2.txt"
val_dataset_path=/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/llm_data/ref_data_refined.jsonl
red="\e[31m"
reset="\e[0m"

#  生成submission文件
echo -e "${red}Generating submission file...${reset}"
echo -e "${red}Submission file created at: \n$submit_jsonl_path${reset}"
python3 generate_submission_from_factory_fuse.py --original_submission_path1 $original_submission_path1 --factory_pred_path1 $factory_pred_path1 --factory_pred_path2 $factory_pred_path2 \
--generated_submission_path $submit_jsonl_path --candidate_file_path $candidate_file_path 


cd /home/zhangtaiyan/workspace/comp/LinkBERT
python gen_cdr.py --test_file $candidate_file_path --validate_file $val_dataset_path 
cd /home/zhangtaiyan/workspace/comp/LinkBERT/src
export MODEL=BioLinkBERT-large2
export MODEL_PATH=/home/zhangtaiyan/workspace/comp/LinkBERT/src/runs/cdr_hf/BioLinkBERT-large2
# export MODEL_PATH=/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/data/biolinkbert_large
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0
task=cdr_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH --candidate_file_path $candidate_file_path \
  --train_file $datadir/train.json --validation_file $datadir/dev1.json --test_file $datadir/test1.json \
  --do_eval --do_predict --metric_name PRF1  --task $task \
  --per_device_train_batch_size 32  --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \


cd /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory
dataset='eval_all'
finetuning_type='full'
cd /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory
model_name_or_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-7B/pt/sft/full/2024-03-29-17-53-22-wo_A-1e-6"


# current_time=$(date +"%Y-%m-%d-%H-%M-%S")
# eval_device=0
# CUDA_VISIBLE_DEVICES=$eval_device python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path ${model_name_or_path} \
#     --finetuning_type ${finetuning_type} \
#     --template default \
#     --dataset_dir data \
#     --dataset ${dataset} \
#     --cutoff_len 8192 \
#     --max_samples 100000 \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate True \
#     --max_new_tokens 256 \
#     --top_p 0.5 \
#     --fp16 True \
#     --temperature 0.5 \
#     --output_dir ${model_name_or_path} \
#     --do_predict True 

# transform format and calculate score



cd /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data
        # 词嵌入模型地址
export MODEL_PATH="/home/zhangtaiyan/workspace/comp/used_data/GoogleNews-vectors-negative300.bin"

# 原始submission文件地址
original_submission_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/submission.jsonl"
# factory生成的预测文件地址
submission1_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/fuse_submission/golden/submission.jsonl"
submission2_path="/home/zhangtaiyan/workspace/comp/LinkBERT/submission.jsonl"
# 生成的submission文件地址
submit_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/fuse_submission/task2_submission.jsonl"
# submit_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/BioMistral-7B/pt/sft/full/2024-04-05-00-02-39-wo_A-1e-6/submission_rep.jsonl"
# 答案文件地址
answer_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data.jsonl"
refined_answer_jsonl_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data_refined.jsonl"
# 分数报告地址
score_report_path="$(dirname "$factory_pred_path")/score.txt"
refined_score_report_path="/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/fuse_submission/refined_score_task2_stanza.txt"

red="\e[31m"
reset="\e[0m"
# 评分
echo -e "${red}Generating submission file...${reset}"
echo -e "${red}Submission file created at: \n$submit_jsonl_path${reset}"
python3 generate_fuse_submission.py --original_submission_path $original_submission_path \
--submission1_path $submission1_path --submission2_path $submission2_path --generated_submission_path $submit_jsonl_path

echo -e "${red}Evaluating...${reset}"
# echo -e "${red}Score report created at: \n$score_report_path${reset}"
# python3 evaluate.py --answer_jsonl_path $answer_jsonl_path --submit_jsonl_path $submit_jsonl_path > $score_report_path
echo -e "${red}Refined core report created at: \n$refined_score_report_path${reset}"
python3 evaluate_refined.py --answer_jsonl_path $refined_answer_jsonl_path --submit_jsonl_path $submit_jsonl_path > $refined_score_report_path

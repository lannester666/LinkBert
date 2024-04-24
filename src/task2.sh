cd MY_PATH/comp/LinkBERT/src
export MODEL=BioLinkBERT-large2
# export MODEL_PATH=MY_PATH/comp/LinkBERT/src/runs/cdr_hf/BioLinkBERT-large2
export MODEL_PATH=MY_PATH/data/biolinkbert_large
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0
task=cdr_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH --candidate_file_path $candidate_file_path \
  --train_file $datadir/train.json --validation_file $datadir/dev1.json --test_file $datadir/test1.json \
  --do_train --do_eval --do_predict --metric_name PRF1  --task $task \
  --per_device_train_batch_size 32  --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \


cd MY_PATH/comp/my_finetune/LLaMA-Factory
dataset='eval_all'
finetuning_type='full'
cd MY_PATH/comp/my_finetune/LLaMA-Factory
model_name_or_path="MY_PATH/comp/my_finetune/LLaMA-Factory/save/Mistral-7B/pt/sft/full/2024-03-29-17-53-22-wo_A-1e-6"

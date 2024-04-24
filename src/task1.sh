cd MY_PATH/comp/LinkBERT/src
export MODEL=BioLinkBERT-large_match_full
# export MODEL_PATH=MY_PATH/comp/LinkBERT/src/runs/agac_hf/BioLinkBERT-large_wo_A_stanza
export MODEL_PATH=MY_PATH/data/biolinkbert_large
export CUDA_VISIBLE_DEVICES=0,2,3,5,6,7
task=agac_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH --candidate_file_path $candidate_file_path \
  --train_file $datadir/train.json --validation_file $datadir/dev1.json --test_file $datadir/test1.json \
  --do_train --do_predict --metric_name PRF1 --task $task \
  --do_eval --per_device_train_batch_size 8 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 25 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \



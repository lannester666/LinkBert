
export MODEL=BioLinkBERT-large
export MODEL_PATH=/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/data/biolinkbert_large
task=DDI_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 3 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  # |& tee $outdir/log.txt &
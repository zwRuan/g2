source activate g2
export CUDA_VISIBLE_DEVICES=0
temperature=1.0
top_p=1.0
top_k=50
min_p=0
filename=temp_sample
python -m eval.xlsum.run_sample \
        --save_dir results/xlsum/baseline/${filename}/base_temp${temperature}_topk${top_k}_topp${top_p}_minp${min_p} \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --temperature $temperature \
        --eval_batch_size 5 \
        --top_k $top_k \
        --top_p $top_p \
        --min_p $min_p \
        --pos_or_neg base \
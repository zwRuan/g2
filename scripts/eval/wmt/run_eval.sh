export CUDA_VISIBLE_DEVICES=0
theta=0.3
temperature=1.0
python -m eval.wmt.run_eval \
        --save_dir results/wmt/theta${theta}_temp${temperature} \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --eval_batch_size 10 \
        --temperature $temperature \
        --theta $theta 

temperature=1.0
weight_method="entropy"
theta=0.3
top_p=1.0
top_k=50

python -m eval.xlsum.run_eval \
        --save_dir results/xlsum/theta${theta}_temp${temperature} \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --eval_batch_size 5 \
        --temperature $temperature \
        --theta ${theta} \
        --alpha 0.5 \
        --weight_method $weight_method \
        --top_p $top_p \
        --top_k $top_k 
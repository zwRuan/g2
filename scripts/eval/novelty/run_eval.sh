iter_num=10
temperature=1
theta=$1
python_file=run_eval.py
outputfile=results/novelty/g2_theta${theta}_temp${temperature}
model=meta-llama/Meta-Llama-3-8B-Instruct
weight_method=entropy
python eval/novelty-bench/src/${python_file} --model_name_or_path $model --data curated --top_k 50 --top_p 1.0 --eval-dir $outputfile --iter_num $iter_num --temperature $temperature --theta $theta --alpha $theta --weight_method $weight_method

source activate eval_g2
python eval/novelty-bench/src/partition.py --eval-dir $outputfile --alg classifier
python eval/novelty-bench/src/score.py --eval-dir $outputfile --patience 0.8
python eval/novelty-bench/src/summarize.py --eval-dir $outputfile
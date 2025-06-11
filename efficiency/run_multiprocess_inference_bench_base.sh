export CUDA_VISIBLE_DEVICES=0 && python multiprocess_bench.py --model Alibaba-NLP/gte-base-en-v1.5 > gte_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && python multiprocess_bench.py --model ModernBERT/bert24-base-v2-learning-rate-decay-v2-50B-3-best-and-last-avg > bert24_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2 && python multiprocess_bench.py --model bert-base-uncased > bert_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3 && python multiprocess_bench.py --model roberta-base > roberta_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=4 && python multiprocess_bench.py --model microsoft/deberta-v3-base > debertav3_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5 && python multiprocess_bench.py --model nomic-ai/nomic-bert-2048 > nomicbert_inference_times.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6 && python multiprocess_bench.py --model Alibaba-NLP/gte-base-en-v1.5 --xformers > gte_xformers_inference_times.log 2>&1 &

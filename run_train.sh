./clear.sh
python3 rm_weights_i_logs.py 0
export CUDA_VISIBLE_DEVICES=1
python3 main_torch_latest.py 0

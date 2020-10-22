export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python mytrain.py \
    --use_dataset CUB \
    --instances 3 \
    --use_loss triplet \
    --lr 0.5e-5 \
    --lr_p 0.25e-5 \
    --lr_gamma 0.1 \
    --sec_wei 1.0 

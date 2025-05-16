DEVICE=$1
PORT=$2

export MASTER_PORT=$PORT

CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/home2/jaehoon/aca2025/practice2-1/DLRM/data/train.txt \
    --processed-data-file=/home2/jaehoon/aca2025/practice2-1/DLRM/data/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --nepochs=1 \
    --print-time \
    --print-freq=1024 \
    --test-freq=1024 \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --memory-map \
    --use-gpu \
    --dist-backend=nccl \
    --tensor-board-filename="run_kaggle_single_gpu"
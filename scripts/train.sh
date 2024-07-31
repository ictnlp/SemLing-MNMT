checkpoints=/path/to/your/checkpoints
setting=/your/setting
lang_pairs='de-en,en-de,en-it,en-nl,en-ro,it-en,nl-en,ro-en'
langs='de,en,it,nl,ro'
datapath=/path/to/your/data

mkdir -p $checkpoints

# train
CUDA_VISIBLE_DEVICES=0,1,2,3  fairseq-train \
    $datapath \
    --tensorboard-logdir $checkpoints \
    --fp16 --seed 1 \
    --ddp-backend=no_c10d \
    --dataset-impl mmap  \
    --dropout 0.3 --attention-dropout 0. \
    --disentangler-lambda 0.05 --disentangler-negative-lambda 0.2 --disentangler-reconstruction-lambda 0.2 \
    --encoder-layers 6 --linguistic-encoder-layers 2 --decoder-layers 6 \
    --share-all-embeddings \
    --task translation_multi_simple_epoch \
    --arch transformer_with_disentangler_and_linguistic_encoder \
    --sampling-method "temperature" --sampling-temperature 5 \
    --encoder-langtok "src" --decoder-langtok \
    --lang-pairs $lang_pairs --langs $langs \
    --left-pad-source False --left-pad-target False \
    --criterion label_smoothed_cross_entropy_with_disentangling --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 7e-4 --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-tokens 2048  --update-freq 2 --max-update 250000 \
    --save-interval 1 --no-epoch-checkpoints \
    --save-interval-updates 1000 --keep-interval-updates 50 \
    --no-progress-bar --log-format json --log-interval 25 \
    --save-dir $checkpoints 2>&1 | tee $checkpoints/out.$setting
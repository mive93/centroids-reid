python train_ctl_model.py \
--config_file="configs/256_resnet18.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR 'data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet18' \
DATALOADER.USE_RESAMPLING False \
# USE_MIXED_PRECISION False \
# TEST.ONLY_TEST True MODEL.PRETRAIN_PATH "models/market1501_resnet50_256_128_epoch_120.ckpt"
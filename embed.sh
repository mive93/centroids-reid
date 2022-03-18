python inference/create_embeddings.py \
--config_file="configs/256_resnet18.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR 'data/ugv/' \
TEST.IMS_PER_BATCH 8 \
OUTPUT_DIR 'outputs_resnet18/' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/micaela/repos/centroids-reid/logs/market1501/256_resnet18/train_ctl_model/version_7/checkpoints/epoch=119.ckpt"

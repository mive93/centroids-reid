python inference/get_similar.py \
--config_file="configs/256_resnet18.yml" \
--gallery_data='outputs_resnet18' \
--normalize_features \
--topk=100 \
GPU_IDS [0] \
DATASETS.ROOT_DIR 'data/ugv/'  \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR 'outputs_resnet18' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/micaela/repos/centroids-reid/logs/market1501/256_resnet18/train_ctl_model/version_7/checkpoints/epoch=119.ckpt" \
SOLVER.DISTANCE_FUNC 'cosine'
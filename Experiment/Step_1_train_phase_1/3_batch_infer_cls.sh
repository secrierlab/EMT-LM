#!/bin/bash

# Define paths and parameters
CODE_LOC="/home/shi/WorkSpace/projects/scMultiNet_workspace/" # path to repo end/with/scMultiNet_project/
DATA_LOC="/home/shi/WorkSpace/projects/scMultiNet_Data/" # path to data folder
RAW_DATA_LOC1=${DATA_LOC}"/Step_1_data/Dataset_Cenk_Q/TrVal_dataset_QuiescenceStatus.pkl" #/home/shi/WorkSpace/projects/scMultiNet_Data/Step_1_data/Dataset_all_counts_inhibitor/TrVal_dataset_Ground_Truth.pkl

echo "RAW_DATA_LOC1: " $RAW_DATA_LOC1

VOCAB_LOC=${CODE_LOC}+"/Experiment/support_data/vocab_16k.json"

echo "VOCAB_LOC: " $VOCAB_LOC

# model_check_points after 1_batch_train.sh 定义模型检查点和输出位置
MODEL_CKPT_Multi=(
${DATA_LOC}"/pretrained/EMT_MultiNet/Cook00_epoch=03-auroc_val=0.87.ckpt"
${DATA_LOC}"/pretrained/EMT_MultiNet/Cook11_epoch=04-auroc_val=0.95.ckpt"
${DATA_LOC}"/pretrained/EMT_MultiNet/Cook22_epoch=04-auroc_val=0.90.ckpt"
${DATA_LOC}"/pretrained/EMT_MultiNet/Cook33_epoch=09-auroc_val=0.92.ckpt"
${DATA_LOC}"/pretrained/EMT_MultiNet/Cook44_epoch=15-auroc_val=0.94.ckpt"
)


VOCAB_PARAMS=${CODE_LOC}"/Experiment/support_data/gene2vec_16906_200.npy"

OUT_FOLDER=${DATA_LOC}"/Step_1_data/Dataset_Cenk_Q/"

echo "OUT_FOLDER: " $OUT_FOLDER

OUT_LOC_Multi_train=(
${OUT_FOLDER}"/cls/cls0.pkl"
${OUT_FOLDER}"/cls/cls1.pkl"
${OUT_FOLDER}"/cls/cls2.pkl"
${OUT_FOLDER}"/cls/cls3.pkl"
${OUT_FOLDER}"/cls/cls4.pkl"
)

OUT_LOC_Multi_val=(
${OUT_FOLDER}"/val/cls0.pkl"
${OUT_FOLDER}"/val/cls1.pkl"
${OUT_FOLDER}"/val/cls2.pkl"
${OUT_FOLDER}"/val/cls3.pkl"
${OUT_FOLDER}"/val/cls4.pkl"
)

# 循环执行推理任务
for i in {0..4}
do
    python scMulti_binary_infer_cls.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt ${MODEL_CKPT_Multi[i]} --vocab_params $VOCAB_PARAMS --out_loc ${OUT_LOC_Multi_train[i]} --index_label "target_label" --train_phase "train"
    #python scMulti_binary_infer.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt ${MODEL_CKPT_Multi[i]} --vocab_params $VOCAB_PARAMS --out_loc ${OUT_LOC_Multi_val[i]} --index_label "target_label" --train_phase "val"
done
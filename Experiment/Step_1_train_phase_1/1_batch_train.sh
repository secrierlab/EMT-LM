
cd "Path/to/repo/Experiment/Step_1_train_phase_1/"

TASK_NAME="EMT_scMultiNet_"
CODE_LOC="Path/to/repo//" # should be something like "/projects/scMultiNet_project/"
RAW_DATA_LOC="/path/to/dataset/saved/in/step_0/TrVal_dataset_PC_TGFb_GTlabel5.pkl"


VOCAB_LOC="Path/to/repo//Experiment/support_data/vocab_16k.json"
VOCAB_PARAMS="Path/to/repo//Experiment/support_data/gene2vec_16906_200.npy"

MODEL_CKPT="Path/to/repo//Experiment/support_data/panglao_pretrain.pth"


OUT_LOC="Path/to/Output_folder/"

for BINARIZE_TARGET in {0..4}; do
    echo "BINARIZE_TARGET: $BINARIZE_TARGET"
    NEW_TASK_NAME="${TASK_NAME}${BINARIZE_TARGET}"
    #python scBERT_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
    python scMulti_binary_Train.py --task_name $NEW_TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
done

#BINARIZE_TARGET=1
#python scBERT_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
#python scBERT_mut_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET



SUB_DROPOUT=0.3
RANDOM=$$
MODEL_CLS=spruce
NUM_LYRS=1
SUB_NUM_LYRS=8
EPOCH=10
GATE=hierarchy_gate
MODE=add
MODEL_NAME=bert-base-uncased
EMB_DIM=768
CTX_AND_COMBINED_MAX_SEQ_LENGTH=96
CTX_AND_COMBINED_TRAIN_BATCH_SIZE=48
SUB_TRAIN_BATCH_SIZE=64


#ctx
python3 train.py \
   --model_cls bert \
   --bert_model $MODEL_NAME \
   --output_dir $CONTEXT_OUTPUT_DIR \
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --num_train_epochs 5 \
   --emb_dim $EMB_DIM \
   --max_seq_length $CTX_AND_COMBINED_MAX_SEQ_LENGTH \
   --mode context \
   --train_batch_size $CTX_AND_COMBINED_TRAIN_BATCH_SIZE \
   --no_finetuning \
   --smin 4 \
   --smax 32 \
   --seed $RANDOM

#sub
python3 train.py \
   --model_cls bert \
   --bert_model $MODEL_NAME \
   --output_dir $FORM_OUTPUT_DIR \
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --num_train_epochs 20 \
   --emb_dim $EMB_DIM \
   --train_batch_size $SUB_TRAIN_BATCH_SIZE \
   --smin 1 \
   --smax 1 \
   --max_seq_length 10 \
   --mode form \
   --learning_rate 0.01 \
   --dropout $SUB_DROPOUT \
   --seed $RANDOM

#fuse models
python3 fuse_models_variants.py \
   --form_model $FORM_OUTPUT_DIR \
   --context_model $CONTEXT_OUTPUT_DIR \
   --mode $MODE \
   --output $FUSED_DIR \
   --variant_model_type $MODEL_CLS \
   --gate_combiner $GATE \
   --variant_num_layers $NUM_LYRS \
   --variant_sub_num_layers $SUB_NUM_LYRS


#train full model
python3 train_variants.py \
   --model_cls $MODEL_CLS \
   --bert_model $FUSED_DIR \
   --output_dir $OUTPUT_DIR \
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --emb_dim $EMB_DIM \
   --mode $MODE \
   --train_batch_size $CTX_AND_COMBINED_TRAIN_BATCH_SIZE \
   --max_seq_length $CTX_AND_COMBINED_MAX_SEQ_LENGTH \
   --num_train_epochs $EPOCH \
   --smin 4 \
   --smax 32 \
   --optimize_only_combinator  \
   --seed $RANDOM \
   --cache_dir /scratch/rpatel17/BERTRAM_2023_OUT/ \
   --learning_rate $LR \
   --dropout $SUB_DROPOUT \

   

#infer vectors
python3 infer_vectors_fixed.py \
   --model $OUTPUT_DIR \
   --input $INPUT_FILE_NAME \
   --output ${OUTPUT_DIR}outputted_embs.txt \
   





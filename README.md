# SPRUCE

To see how this code is run, see train_bertram_on_pca_embs.sh .

SPRUCE model defined in bertram_variants.py

This code is based on the code from https://github.com/timoschick/bertram

Here is a summary on how to use:

1) Build a corpus using preprocess from https://github.com/timoschick/form-context-model.
2) Train a context mode and subword model using commands in sh script (create directory vars).
3) Fuse the models (see sh script).
4) Run full model (see sh script).
5) To use model to estimate rare words (for whichever task), use preprocess from https://github.com/timoschick/form-context-model on task corpus and list of rare words, then call infer_vectors_fixed.
6) Use BERTRAMWrapper (for details, see bertram.py and https://github.com/timoschick/bertram) to use estimated rare embeddings in final task.


For the evaluation tasks, please refer to the sources cited in the paper.

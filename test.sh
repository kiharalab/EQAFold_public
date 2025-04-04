python run.py \
    --batch 1\
    --device_id 0\
    --model alphafold\
    --num_workers 1\
    --msa_transformer_dir ./example/example_embeddings \
    --af2\
    --test_targets ./example/example.list \
    --contact_cutoff 16.0 \
    --graph_type EGNN \
    --lddt_weight 1.0 \
    --output_dir ./example/example_output\
    --model_dir  ./safetensors/checkpoint-52-39644\
    --edge_feats \
    --esm_feats\
    --rmsf_feats\
    --esm_edgefeats_alllayer\
    --test_only
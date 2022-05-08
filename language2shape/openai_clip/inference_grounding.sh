#!/bin/bash
map_ckpt="$1"
nf_ckpt="$2"

python openai_clip/inference_grounding.py \
        --cate "chair" \
        --from_txt \
        --split_by_id \
        --no_use_text2shape \
        --clip_txt_feat_path 'data/clip_txt_feat_text2shape.pickle' \
        --text "a cuboid sofa;a round sofa;a circular bench;a cuboid bench;a round bench;a thick bench;a thin bench;a stool;a round chair;a square chair;a cuboid chair;a long chair;a short chair;a tall chair;an armchair;a chair with long legs;a chair has long legs;a chair with short legs;a chair has short legs;a chair with a back;a chair without back;a chair with a round back;a chair with a cuboid back;a star-shaped chair;a cushioned arm chair;a large cushioned recliner chair;a wooden rocking chair;a small ladder back chair;a kid’s high chair;a high chair for kids;a short stool chair;a tall office swivel chair with x wheels;a tall office swivel chair;an office chair with headrest;a cushioned sofa;a round stool chair"\
        --use_txt_prefixs \
        --valid_cate_anno_path "../../../data/ShapeNetCore.v1/meta/03001627" \
        --finetune_clip \
        --map_code_base 'B_MLPS' \
        --map_restore_file "openai_clip/finetune/results/165196138154665"\
        --norm 'LN'\
        --act_f 'tanh'\
        --l2norm \
        --num_layers 1\
        --h_features 512 \
        --nf_code_base 'pytorch_flows' \
        --nf_model 'realnvp' \
        --nf_restore_file "pytorch_flows/results/pytorch_flows/results/165196138154665_1651962355445" \
        --cond_label_size 4096 \
        --n_blocks 10 \
        --hidden_size 2048\
        --input_size 512 \
        --use_mlp\
        --n_row 5\
        # --n_from_img 0 \
        # --img_gt_path "../../../data/render_ShapeNetCore.v1/03001627/1view_texture_ground_keyshot_4secs" \
        # --shape_gt_path "occupancy_networks/data/ShapeNet/03001627" \
        # --embed_feat_path "occupancy_networks/out/pointcloud/onet/pretrained/embed/03001627/embed_feats_train.pickle" \
        # --embed_feat_test_path "occupancy_networks/out/pointcloud/onet/pretrained/embed/03001627/embed_feats_val.pickle" \
        # --out_mesh_folder 'occupancy_networks/out/pointcloud/onet/pretrained/generation/meshes_from_z/conditional/'$2 \
        # --quality \
        # --clip_img_feat_path 'data/clip_img_feat_1view_texture_ground_keyshot.pickle'\
        # --n_retrieval_clip 0\
        # --n_retrieval_shape 1 \
        # --quantity \
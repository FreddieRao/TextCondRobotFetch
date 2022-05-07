python openai_clip/finetune_clip.py \
        --clip_img_feat_path "../../data/embed_ShapeNetCore.v1/03001627/clip_img_feat_1view_texture_ground_keyshot.pickle" \
        --clip_txt_feat_path "../../data/embed_ShapeNetCore.v1/03001627/clip_txt_feat_text2shape.pickle" \
        --loss_name_list scontrast \
        --loss_weights 1
python openai_clip/finetune_clip.py \
        --clip_img_feat_path "data/clip_img_feat_1view_texture_ground_keyshot.pickle" \
        --clip_txt_feat_path "data/clip_txt_feat_text2shape.pickle" \
        --loss_name_list scontrast_cos_disdiv \
        --loss_weights 0.9_0.09_0.01 
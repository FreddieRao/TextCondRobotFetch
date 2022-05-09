python inference.py --dataset sunrgbd \
--checkpoint_path demo_files/pretrained_votenet_on_sunrgbd.tar \
--dump_dir results --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
--pc_path black_chair/pc_batch000004_item_000_cls_3_npc_00338.ply
# --per_class_proposal
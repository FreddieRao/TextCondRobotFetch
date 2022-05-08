python inference.py --dataset sunrgbd \
--checkpoint_path demo_files/pretrained_votenet_on_sunrgbd.tar \
--dump_dir inference_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms 
# --per_class_proposal
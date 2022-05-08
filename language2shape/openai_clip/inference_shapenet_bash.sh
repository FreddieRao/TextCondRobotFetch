#!/bin/bash
map_ckpt="$1"
nf_ckpt="$2"

python openai_clip/inference_shapenet.py \
        --cate "chair" \
        --from_txt \
        --split_by_id \
        --no_use_text2shape \
        --clip_txt_feat_path 'data/clip_txt_feat_text2shape.pickle' \
        --text "a cuboid sofa;a round sofa;a circular bench;a cuboid bench;a round bench;a thick bench;a thin bench;a stool;a round chair;a square chair;a cuboid chair;a long chair;a short chair;a tall chair;an armchair;a chair with long legs;a chair has long legs;a chair with short legs;a chair has short legs;a chair with a back;a chair without back;a chair with a round back;a chair with a cuboid back;a star-shaped chair;a cushioned arm chair;a large cushioned recliner chair;a wooden rocking chair;a small ladder back chair;a kid’s high chair;a high chair for kids;a short stool chair;a tall office swivel chair with x wheels;a tall office swivel chair;an office chair with headrest;a cushioned sofa;a round stool chair"\
        --use_txt_prefixs \
        --valid_cate_anno_path "../../../data/ShapeNetCore.v1/meta/03001627" \
        --n_from_img 0 \
        --img_gt_path "../../../data/render_ShapeNetCore.v1/03001627/1view_texture_ground_keyshot_4secs" \
        --clip_img_feat_path 'data/clip_img_feat_1view_texture_ground_keyshot.pickle'\
        --n_retrieval_clip 0\
        --n_retrieval_shape 1 \
        --quantity \
        --finetune_clip \
        --map_code_base 'B_MLPS' \
        --map_restore_file "openai_clip/finetune/results/"$1\
        --norm 'LN'\
        --act_f 'tanh'\
        --l2norm \
        --num_layers 1\
        --h_features 512 \
        --nf_code_base 'pytorch_flows' \
        --nf_model 'realnvp' \
        --nf_restore_file "pytorch_flows/results/pytorch_flows/results/"$2 \
        --cond_label_size 4096 \
        --n_blocks 10 \
        --hidden_size 2048\
        --input_size 512 \
        --use_mlp\
        --shape_gt_path "occupancy_networks/data/ShapeNet/03001627" \
        --embed_feat_path "occupancy_networks/out/pointcloud/onet/pretrained/embed/03001627/embed_feats_train.pickle" \
        --embed_feat_test_path "occupancy_networks/out/pointcloud/onet/pretrained/embed/03001627/embed_feats_val.pickle" \
        --out_mesh_folder 'occupancy_networks/out/pointcloud/onet/pretrained/generation/meshes_from_z/conditional/'$2 \
        --n_row 1\
        --quality \
        # --post_process $4 \
        # --vis_correct_only \
        # --vis_txt_embed \
        # --vis_img_embed \
        # --no_batch_norm \
        # --unconditional
# 03001627 03001627
# free-formed chair "a cuboid sofa;a round sofa;a circular bench;a cuboid bench;a round bench;a thick bench;a thin bench;a stool;a round chair;a square chair;a cuboid chair;a long chair;a short chair;a tall chair;an armchair;a chair with long legs;a chair has long legs;a chair with short legs;a chair has short legs;a chair with a back;a chair without back;a chair with a round back;a chair with a cuboid back;a star-shaped chair;a cushioned arm chair;a large cushioned recliner chair;a wooden rocking chair;a small ladder back chair;a kid’s high chair;a high chair for kids;a short stool chair;a tall office swivel chair with x wheels;a tall office swivel chair;an office chair with headrest;a cushioned sofa;a round stool chair"
# free-formed chair "a cushioned arm chair;a large cushioned recliner chair;a wooden rocking chair;a small ladder back chair;a kid’s high chair;a high chair for kids;a short stool chair;a tall office swivel chair with x wheels;a tall office swivel chair;an office chair with headrest;a cushioned sofa;a round stool chair"
# free-formed table "an oval table;a rectangular table;a table with a round surface and a round base;a table with a square surface and a round base;a table has circular surface and four straight legs;a table has rectangular surface and four straight legs;a tall table;a short table;a long table;a multi-layered table;a two-layered table;a table with a drawer;a table with storage;a table with a single pedestal;a table with two legs;a table with two broad legs;a table with four legs;a table with four decorative legs;a shelf;a compuater desk"
# free-formed table "a short nightstand with two drawers;a drawing table with two legs;a short coffee table with two layers and three legs;a short coffee table with two layers;a short coffee table with three legs;a tall accent table with crossed legs;a plastic outdoor table with four legs;a square dining table with four legs;a round conference table;a tall vanity table with one drawer"
# bash openai_clip/inference_shapenet_bash.sh 0000 None None l2norm 0 0
# chair text "ball chair;cantilever chair;armchair;tulip chair;straight chair;side chair;club chair;swivel chair;easy chair;lounge chair;overstuffed chair;barcelona chair;chaise longue;chaise;daybed;deck chair;beach chair;folding chair;bean chair;butterfly chair;rocking chair;rocker;zigzag chair;recliner;reclining chair;lounger;lawn chair;garden chair;Eames chair;sofa;couch;lounge;rex chair;camp chair;X chair;Morris chair;NO. 14 chair;park bench;table;wassily chair;Windsor chair;love seat;loveseat;tete-a-tete;vis-a-vis;wheelchair;bench;wing chair;ladder-back;ladder-back chair;throne;double couch;settee"
# car text "convertible;racer;race car;racing car;roadster;runabout;two-seater;coupe;stock car;touring car;phaeton;tourer;sport utility;sport utility vehicle;S.U.V.;SUV;sedan;saloon;cruiser;police cruiser;patrol car;police car;prowl car;squad car;beach wagon;station wagon;wagon;estate car;beach waggon;station waggon;waggon;hot rod;hot-rod;pace car;limousine;limo;jeep;landrover;hatchback;cab;hack;taxi;taxicab;minivan;sports car;sport car;ambulance"
# table text "desk cabinet;table;desk;side table;cabinet table;console table;console;drafting table;drawing table;worktable;work table;counter;coffee table;cocktail table;secretary;writing table;escritoire;secretaire;writing desk;kitchen table;park bench;bench;pool table;billiard table;snooker table;lab bench;laboratory bench;table-tennis table;ping-pong table;pingpong table;sofa;couch;lounge;pedestal table;tea table;drop-leaf table;file;file cabinet;filing cabinet;card table;lectern;reading desk;rectangular table;conference table;council table;council board;round table;bar;short table;chair;workshop table;stand;reception desk;altar;communion table;Lord's table;soda fountain;checkout;checkout counter;operating table;tilt-top table;tip-top table;tip table;dressing table;dresser;vanity;toilet table"        
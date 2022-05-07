file_name="$1"
cate="$2"
python pytorch_flows/main.py --dataset 'ClipImage2ShapeCache' \
                            --flow 'realnvp' \
                            --cond \
                            --num-blocks 10 \
                            --use_mlps \
                            --lr 0.0001 \
                            --no_finetune_clip \
                            --map_restore_file $file_name \
                            --cate $cate


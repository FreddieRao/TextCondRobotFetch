# Conditional Shape Generation
Script code for Conditional Shape Generation

## OCNET
For training and generating OCNET features, please run scripts:
```
bash occupancy_networks/train_bash.sh
bash occupancy_networks/gen_bash.sh
```

## Openai-CLIP
For generating CLIP features, please run scripts:
```
python openai_clip/preprocess_shapenet.py 
```

For generating shape with trained OCNET and NF, please run scripts:
```
bash openai_clip/inference_realnvp_bash.sh
```

## Normalize Flow(RealNVP Normalizing Flows codebase)
For training and inferencing Noralize Flow with trained OCNET and CLIP features, please run scripts:
```
cd 
bash normalizing_flows/train_bash.sh
bash normalizing_flows/inference_bash.sh
```
## Normalize Flow(RealNVP Pytorch Flows codebase)
For training Noralize Flow with trained OCNET and CLIP features, please run scripts:
```
cd 
bash pytorch_flows/train_bash.sh
```

## Evaluation
Visualize a folder of mesh and render a folder of mesh with their file name:

```
bash openai_clip/inference_shapenet_bash.sh
```

## Visualization
Visualize a folder of mesh and render a folder of mesh with their file name:

```
cd utils
python result_visualization.py 
# arguments: input_folder, n_rows, save_mesh, save_img
```
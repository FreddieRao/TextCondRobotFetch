# Text-condition Object Detection with Point Cloud
The repo comprises three parts: Detect shapes in point clouds with votenet, Text-conditioned Shape generation with CLIP-Forge similar structure, and Shape similarity caluculation with OCNET.

## Shape Detection
Shape Detection with VoteNet:
```
cd votenet_detection
bash inference.sh

# change --pc_path to change path to input point cloud
# change --dump_dir to chage the directory to save results
```
Which will return oriented pointclouds of objects in the scene.

## Text-conditioned Shape Generation
Please run scripts:
```
cd language2shape
bash openai_clip/inference_grounding.sh
# change --text to change text input; --n_row number of shape features; --dump_dir directory to save shape features and texts
```
Which will return a list of features of text description.

## Shape Similarity
Calculate similarity between detected shapes and text-conditioned generated shapes, please run:

```
cd shapesimilarity
bash shape_similarity.sh
```

## TODO
-  With Scanned input point cloud, extract object with Shape Detection Part.
-  Caculate Similarity Between Shapes and text-conditioned shape features.
- [ ] Run a inference script for (1) point-cloud->shape list; (2)text->shape feautures; (3) find most similar shape in the shape list->output bbx.
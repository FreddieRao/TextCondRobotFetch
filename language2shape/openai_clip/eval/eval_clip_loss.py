import os
import torch
import numpy as np
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def calculate_clip_loss(image_token, text_token, model=model):
    with torch.no_grad():
        image_features = model.encode_image(image_token)
        text_features = model.encode_text(text_token)
        dist = torch.norm(image_features - text_features)
    return dist

def eval_img_folder(input_folder):
    f_name_list = [ _ for _ in os.listdir(input_folder) if (_.endswith('png') or _.endswith('jpg'))]
    f_name_list = [ _ for _ in f_name_list if (('rtv' not in _) and ('gt' not in _) and ('image_list' not in _) )]
    total_loss = 0.
    for f_name in f_name_list:
        f_path = os.path.join(input_folder, f_name)
        image = preprocess(Image.open(f_path)).unsqueeze(0).to(device)
        text = f_name.split('.')[0].replace('_', '')
        text = clip.tokenize([text]).to(device)
        total_loss += calculate_clip_loss(image, text)
    avg_loss = total_loss / len(f_name_list)
    avg_loss = np.asarray(avg_loss.to('cpu'))
    print("Average clip loss is {} on {}".format(avg_loss, input_folder))
    return avg_loss

if __name__ == '__main__':
    input_folder = './occupancy_networks/out/voxels/onet_chair/generation/meshes_from_z/conditional/1638898775750'
    eval_img_folder(input_folder)

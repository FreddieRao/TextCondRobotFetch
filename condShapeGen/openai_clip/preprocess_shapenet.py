import os
import numpy as np
import torch
import torchvision
import clip
from PIL import Image
import pickle
import cv2
import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print(preprocess)
preprocess_new = torchvision.transforms.Compose([preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[2]])
# preprocess_new = preprocess.transforms[0]
# exit(0)

def process_shapenet():
    shapenet_root = "occupancy_networks/data/ShapeNet"
    data_root = "occupancy_networks/data/ShapeNet/03001627"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat.pickle')

    embed_feats = {}

    for obj_idx, object_name in enumerate(os.listdir(data_root)):
        print(obj_idx)
        embed_feats[object_name] = {}
        object_dir = os.path.join(data_root, object_name)
        image_dir = os.path.join(object_dir, "img_choy2016")
        if os.path.isdir(image_dir):
            for img_idx, image_name in enumerate(os.listdir(image_dir)):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    image_path = os.path.join(image_dir, image_name)
                    image = Image.open(image_path)
                    image = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                        embed_feats[object_name][str(img_idx).zfill(2)] = image_feat

    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_1view_shade():
    data_root = "../../data/render_ShapeNetCore.v1/04379243/1view_texture_ground_keyshot_4secs"
    shapenet_root = "../../data/embed_ShapeNetCore.v1/04379243"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1view_texture_ground_keyshot.pickle')

    embed_feats = {}
    
    for obj_idx, object_name in tqdm.tqdm(enumerate(os.listdir(data_root))):
        if object_name.endswith('.jpg') or object_name.endswith('.png'):
            embed_feats[object_name.replace('.png', '')] = {}
            image_path = os.path.join(data_root, object_name)
            try:
                image = Image.open(image_path)
                # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                # mask = np.where(image[:, :, 3]==0, True, False)
                # image = image[:, :, 0:3]
                # image[mask, :] = np.array([255, 255, 255])
                # # image = image / 255
                # image = Image.fromarray(image)
                image = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                    embed_feats[object_name.replace('.png', '')][str(0).zfill(2)] = image_feat
            except:
                pass
                
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_1view_shade_cifarbg():
    data_root = "../../data/render_ShapeNetCore.v1/1_view_shade_10bright"
    shapenet_root = "../../data/embed_ShapeNetCore.v1"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1view_shade_10bright.pickle')

    embed_feats = {}
    
    for obj_idx, object_name in enumerate(os.listdir(data_root)):
        print(obj_idx)
        if object_name.endswith('.jpg') or object_name.endswith('.png'):
            obj_name, bg_name = object_name.replace('.png', '').split('_')
            embed_feats[obj_name] = {}
            image_path = os.path.join(data_root, object_name)
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                embed_feats[obj_name][bg_name] = image_feat
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_1view_shade_mean_cifarbg():
    data_root = "../../data/render_ShapeNetCore.v1/1_view_shade_10bright"
    shapenet_root = "../../data/embed_ShapeNetCore.v1"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1view_shade_mean_10bright.pickle')

    embed_feats = {}
    
    for obj_idx, object_name in enumerate(os.listdir(data_root)):
        print(obj_idx)
        if object_name.endswith('.jpg') or object_name.endswith('.png'):
            obj_name, bg_name = object_name.replace('.png', '').split('_')
            if obj_name not in embed_feats.keys():
                embed_feats[obj_name] = {}
                image_feat_list = []
                for view_idx in range(10):
                    try:
                        image_path = os.path.join(data_root, obj_name + '_' + str(view_idx).zfill(3) + '.png')
                        image = Image.open(image_path)
                        image = preprocess(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                            image_feat_list.append(image_feat)
                    except:
                        pass
                embed_feats[obj_name][str(0).zfill(2)] = np.mean(np.stack(image_feat_list, axis=0), axis=0)
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                             
def process_1view_color():
    data_root = "../../../data/render_ShapeNetCore.v1/1_view_ColorTransfer"
    shapenet_root = "../../../data/embed_ShapeNetCore.v1"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1_view_color.pickle')

    embed_feats = {}

    for obj_idx, object_name in enumerate(os.listdir(data_root)):
        print(obj_idx)
        if object_name.endswith('.jpg') or object_name.endswith('.png'):
            embed_feats[object_name.replace('.png', '')] = {}
            image_path = os.path.join(data_root, object_name)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # mask = np.where(image[:, :, 3]==0, True, False)
            # image = image[:, :, 0:3]
            # image[mask, :] = np.array([255, 255, 255])
            # # image = image / 255
            # image = Image.fromarray(image)
            # image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                embed_feats[object_name.replace('.png', '')][str(0).zfill(2)] = image_feat
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_20view_shade():
    data_root = "../../data/render_ShapeNetCore.v1/03001627/1view_texture_ground_keyshot_4secs_f_b6_12"
    shapenet_root = "../../data/embed_ShapeNetCore.v1/03001627"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1view_texture_ground_keyshot_aug.pickle')

    embed_feats = {}

    for obj_idx, object_name in tqdm.tqdm(enumerate(os.listdir(data_root))):
        object_dir_path = os.path.join(data_root, object_name)
        if os.path.isdir(object_dir_path):
            embed_feats[object_name] = {}
            for img_idx, image_name in enumerate(os.listdir(object_dir_path)):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    image_path = os.path.join(object_dir_path, image_name)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                        embed_feats[object_name.replace('.png', '')][str(img_idx).zfill(2)] = image_feat
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_20view_shade_mean():
    data_root = "../../data/render_ShapeNetCore.v1/1_view_shade_10bright"
    shapenet_root = "../../data/embed_ShapeNetCore.v1"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_1view_shade_10bright_mean.pickle')

    embed_feats = {}

    for obj_idx, object_name in tqdm.tqdm(enumerate(os.listdir(data_root))):
        object_dir_path = os.path.join(data_root, object_name)
        if os.path.isdir(object_dir_path):
            if len(os.listdir(object_dir_path)) > 2:
                embed_feats[object_name] = {}
                image_feat_list = []
                for img_idx, image_name in enumerate(os.listdir(object_dir_path)):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        image_path = os.path.join(object_dir_path, image_name)
                        # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        # mask = np.where(image[:, :, 3]==0, True, False)
                        # image = image[:, :, 0:3]
                        # image[mask, :] = np.array([255, 255, 255])
                        # # image = image / 255
                        # image = Image.fromarray(image)
                        # image = preprocess(image).unsqueeze(0).to(device)
                        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                            image_feat_list.append(image_feat)
                embed_feats[object_name.replace('.png', '')][str(0).zfill(2)] = np.mean(np.stack(image_feat_list, axis=0), axis=0)
            else:
                print(object_dir_path)
            
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_20view_color(color_num=5):
    data_root = "../../../data/render_ShapeNetCore.v1/20_view_shade_ColorTransfer"
    shapenet_root = "../../../data/embed_ShapeNetCore.v1"
    img_feat_file = os.path.join(shapenet_root, 'clip_img_feat_20view_color.pickle')

    embed_feats = {}

    for obj_idx, object_name in enumerate(os.listdir(data_root)):
        print(obj_idx)
        object_dir_path = os.path.join(data_root, object_name)
        if os.path.isdir(object_dir_path):
            embed_feats[object_name] = {}
            for image_name in os.listdir(object_dir_path):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    img_idx, n = image_name.replace('.png', '').split('_')
                    img_idx = int(img_idx)
                    n = int(n)
                    image_path = os.path.join(object_dir_path, image_name)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_feat = model.encode_image(image).squeeze(0).cpu().numpy()
                        embed_feats[object_name.replace('.png', '')][str(img_idx).zfill(2)+str(n).zfill(2)] = image_feat
    
    with open(img_feat_file, 'wb') as handle:
        pickle.dump(embed_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # process_shapenet():
    process_20view_shade()
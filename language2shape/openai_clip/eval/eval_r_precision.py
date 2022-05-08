import pandas as pd
import math

# 53 classes
all_valid_name = ['ball chair', 'cantilever chair', 'armchair', 'tulip chair', 'straight chair', 'side chair', 'club chair', 'swivel chair', 'easy chair', 'lounge chair', 'overstuffed chair', 'barcelona chair', 'chaise longue', 'chaise', 'daybed', 'deck chair', 'beach chair', 'folding chair', 'bean chair', 'butterfly chair', 'rocking chair', 'rocker', 'zigzag chair', 'recliner', 'reclining chair', 'lounger', 'lawn chair', 'garden chair', 'Eames chair', 'sofa', 'couch', 'lounge', 'rex chair', 'camp chair', 'X chair', 'Morris chair', 'NO. 14 chair', 'park bench', 'table', 'wassily chair', 'Windsor chair', 'love seat', 'loveseat', 'tete-a-tete', 'vis-a-vis', 'wheelchair', 'bench', 'wing chair', 'ladder-back', 'ladder-back chair', 'throne', 'double couch', 'settee']
file_name = "../../../data/ShapeNetCore.v1/03001627.csv"
df = pd.read_csv(file_name, usecols= ['fullId', 'wnlemmas'])
df = df.reset_index()  # make sure indexes pair with number of rows

def get_all_valid_names(args):
    if args.cate == 'chair':
        tar_df = df

    all_valid_names = {}
    for index, row in tar_df.iterrows():
        valid_names = row['wnlemmas'].split(',')
        for name in valid_names:
            if name not in list(all_valid_names.keys()):
                all_valid_names[name] = 1
            else:
                all_valid_names[name] += 1

    print(len(all_valid_names))

    all_valid_names_1 = [key for key in all_valid_names if all_valid_names[key] > 1 ]
    all_valid_names_str = ''
    for name in all_valid_names_1:
        all_valid_names_str += name + ';'

    print(all_valid_names_1)
    print(len(all_valid_names_1))
    print(all_valid_names_str)
    

def precision_by_txt(args, A_txt, B_id):
    if args.cate == 'chair':
        tar_df = df

    try:
        B_names = tar_df[tar_df.fullId == ('3dw.' + B_id) ]['wnlemmas'].values[0].split(',')
    except:
        return None
    if 'chair' in B_names:
        B_names.remove('chair')
    if 'table' in B_names:
        B_names.remove('table')
        
    if len(B_names)==0:
        # Can't tell the subcategory
        return None
    else:
        return int(A_txt in B_names)

def precision_by_idx(args, A_id, B_id):
    if args.cate == 'chair':
        tar_df = df
    
    A_names = tar_df[tar_df.fullId == ('3dw.' + A_id) ]['wnlemmas'].values[0].split(',')
    try:
        B_names = tar_df[tar_df.fullId == ('3dw.' + B_id) ]['wnlemmas'].values[0].split(',')
    except:
        return None
    if 'chair' in A_names:
        A_names.remove('chair')
    if 'chair' in B_names:
        B_names.remove('chair')
    if 'table' in A_names:
        A_names.remove('table')
    if 'table' in B_names:
        B_names.remove('table')

    if len(A_names)==0 or len(B_names)==0:
        # Can't tell the subcategory
        return None
    else:
        # More than 1/2 of the subcategories match
        least_required = min(math.ceil(len(A_names)/2), math.ceil(len(B_names)/2))
        matches = sum([int(A_name in B_names) for A_name in A_names])
        return int(least_required <= matches)

def precision_of_text2shape(text_str, clip_name_array, idx_sort, full_names, rtv_num=5):
    precision = 0
    precision5 = 0
    for p_idx in range(rtv_num):
        p_clip_name = clip_name_array[idx_sort[p_idx]]
        p_clip_model_name = p_clip_name.split('@')[0]
        p_full_name = [_ for _ in full_names if _.startswith(p_clip_model_name)][0]
        t_name = text_str.split('@')[0]
        if t_name == p_full_name:
            if p_idx == 0:
                precision = 1
            precision5 = 1
    return precision, precision5
            
if __name__ == "__main__":
    get_all_valid_names(df)
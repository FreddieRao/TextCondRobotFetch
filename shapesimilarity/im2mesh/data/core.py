import os
import logging
from torch.utils import data
import numpy as np
import yaml
import random

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, c_idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True

class ShapeNetPairDataset(data.Dataset):
    ''' 3D Shapes dataset: partialscan and complete scane.
    '''

    def __init__(self, complete_dataset_folder= 'data/ShapeNet/03001627', partial_dataset_folder = 'data/PartialShapeNetChair/dataset_small_partial/dataset_small_partial/03001627/03001627', split='train'):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
        '''
        # Attributes
        self.complete_dataset_folder = complete_dataset_folder
        self.partial_dataset_folder = partial_dataset_folder

        split_file = os.path.join(complete_dataset_folder, split + '.lst')
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        
        self.complete_models = []
        self.partial_models = []
        self.unpair_partial_models = []

        valid_model_c = []
        for m in models_c:
            m_complete_folder = os.path.join(complete_dataset_folder, m)
            m_partial_folder = os.path.join(partial_dataset_folder, m)
            if os.path.exists(m_complete_folder) and os.path.exists(m_partial_folder):
                valid_model_c.append(m)
        
        for m_idx in range(len(valid_model_c)):
            m = valid_model_c[m_idx]
            unpair_m = valid_model_c[m_idx-1] if m_idx != 0 else valid_model_c[-1]
            m_complete_folder = os.path.join(complete_dataset_folder, m)
            m_partial_folder = os.path.join(partial_dataset_folder, m)
            m_complete_path = os.path.join(m_complete_folder, 'pointcloud.npz')
            m_partial_paths = []
            for file in os.listdir(m_partial_folder):
                if file.endswith('partial.npz'):
                    m_partial_paths.append(os.path.join(m_partial_folder, file))
        
            unpair_m_partial_folder = os.path.join(partial_dataset_folder, unpair_m)
            unpair_m_partial_paths = []
            for file in os.listdir(unpair_m_partial_folder):
                if file.endswith('partial.npz'):
                    unpair_m_partial_paths.append(os.path.join(unpair_m_partial_folder, file))
            unpair_m_partial_paths = random.choices(unpair_m_partial_paths, k=len(m_partial_paths))
            
            self.complete_models += [m_complete_path] * len(m_partial_paths)
            self.partial_models += m_partial_paths
            self.unpair_partial_models += unpair_m_partial_paths

        print("# Complete Scans:", len(self.complete_models))
        print("# Partial Scans:", len(self.partial_models))
        print("# Unapir Partial Scans:", len(self.unpair_partial_models))

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.complete_models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        complete_model_path = self.complete_models[idx]
        partial_model_path = self.partial_models[idx]
        unpair_partial_model_path = self.unpair_partial_models[idx]
        
        complete_pointcloud_dict = np.load(complete_model_path)
        partial_pointcloud_dict = np.load(partial_model_path)
        unpair_partial_pointcloud_dict = np.load(unpair_partial_model_path)

        complete_points = complete_pointcloud_dict['points'].astype(np.float32)
        partial_points = partial_pointcloud_dict['points_r'].astype(np.float32)
        unpair_partial_points = unpair_partial_pointcloud_dict['points_r'].astype(np.float32)

        indices = np.random.randint(complete_points.shape[0], size=2048)
        complete_points = complete_points[indices, :]
        indices = np.random.randint(partial_points.shape[0], size=2048)
        partial_points = partial_points[indices, :]
        
        complete_points_dict = {
            'inputs': complete_points
        }
        partial_points_dict = {
            'inputs': partial_points
        }
        unpair_partial_points_dict = {
            'inputs': unpair_partial_points
        }
        return complete_points_dict, partial_points_dict, unpair_partial_points_dict


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

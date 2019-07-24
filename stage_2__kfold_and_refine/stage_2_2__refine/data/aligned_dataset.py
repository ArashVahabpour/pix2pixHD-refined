import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        self.dir_A = os.path.join(opt.dataroot, self.opt.phase + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.dataset_size = len(self.A_paths)

        if opt.phase == 'train':
            kf = KFold(n_splits=opt.num_nets, shuffle=True, random_state=42)
            validation_set_list = [list(kf.split(np.arange(self.dataset_size)))[net_idx][1] for net_idx in range(self.opt.num_nets)]
            self.img_idx_to_net_idx = dict(sum([zip([net_idx] * len(validation_set_list[net_idx]),
                                                    validation_set_list[net_idx])
                                                for net_idx in range(self.opt.num_nets)]))

    def __getitem__(self, index):

        def replace_last_occurence(s, old, new):
            """ Replace last occurrence of a string """
            return new.join(s.rsplit(old, 1))

        A_path_ref = self.A_paths[index]  # reference path which will be used for string replacement

        if self.opt.phase == 'train':
            net_idx = self.img_idx_to_net_idx[index]
            raise NotImplementedError('verify the path below according to how val results are stored')
            raise NotImplementedError('merge stages 2_1 and 2_2')
            split_basename = os.path.splitext(os.path.basename(A_path_ref))  ## '/path/to/image/xyz.png' --> ('xyz', 'png')
            A_path = os.path.join('..',
                                  'stage_2_1__kfold',
                                  'results',
                                  net_idx + '.' + str(net_idx),
                                  split_basename[0] + '_synthesized' + split_basename[1])

        else:  # i.e. self.opt.phase == 'test'
            A_path = A_path_ref

        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)[:1]

        D_path = replace_last_occurence(A_path_ref, self.opt.phase + '_A', self.opt.phase + '_D')
        D = Image.open(D_path).convert('RGB')
        transform_D = get_transform(self.opt, params)
        D_tensor = transform_D(D)[:1]

        E_path = replace_last_occurence(A_path_ref, self.opt.phase + '_A', self.opt.phase + '_E')
        E = Image.open(E_path).convert('RGB')
        transform_E = get_transform(self.opt, params)
        E_tensor = transform_E(E)[:1]


        B_tensor = 0
        ### input B (real images)
        if self.opt.isTrain: # or self.opt.use_encoded_image:
            B_path = replace_last_occurence(A_path_ref, self.opt.phase + '_A', self.opt.phase + '_B')
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)[:1]

        input_dict = {'label': torch.cat((A_tensor, D_tensor, E_tensor)),
                      'image': B_tensor, 'path': A_path_ref}

        return input_dict

    def __len__(self):
        return (len(self.kf_indices) if self.use_training_data else self.dataset_size) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

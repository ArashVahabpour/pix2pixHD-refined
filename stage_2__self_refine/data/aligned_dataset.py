import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        def replace_last_occurence(s, old, new):
            """ Replace last occurrence of a string """
            return new.join(s.rsplit(old, 1))

        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)[:1]

        D_path = replace_last_occurence(A_path, self.opt.phase + '_A', self.opt.phase + '_D')
        D = Image.open(D_path).convert('RGB')
        transform_D = get_transform(self.opt, params)
        D_tensor = transform_D(D)[:1]

        E_path = replace_last_occurence(A_path, self.opt.phase + '_A', self.opt.phase + '_E')
        E = Image.open(E_path).convert('RGB')
        transform_E = get_transform(self.opt, params)
        E_tensor = transform_E(E)[:1]


        ### input B, C (real images and edges)
        B_path = replace_last_occurence(A_path, self.opt.phase + '_A', self.opt.phase + '_B')
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)[:1]

        C_path = replace_last_occurence(A_path, self.opt.phase + '_A', self.opt.phase + '_C')
        C = Image.open(C_path).convert('RGB')
        transform_C = get_transform(self.opt, params)
        C_tensor = transform_C(C)[:1]

        label = torch.cat((A_tensor, D_tensor, E_tensor)) if self.opt.with_context else A_tensor

        input_dict = {'label': label,
                      'image': B_tensor, 'edge': C_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

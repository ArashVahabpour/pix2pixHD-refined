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
        self.dir_A = os.path.join(opt.dataroot, 'train_A')  # , opt.phase + dir_A)
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

        D_path = replace_last_occurence(A_path, 'train_A', 'train_D')
        D = Image.open(D_path).convert('RGB')
        transform_D = get_transform(self.opt, params)
        D_tensor = transform_D(D)[:1]

        E_path = replace_last_occurence(A_path, 'train_A', 'train_E')
        E = Image.open(E_path).convert('RGB')
        transform_E = get_transform(self.opt, params)
        E_tensor = transform_E(E)[:1]

        B_tensor = 0
        ### input B (real images)
        if self.opt.isTrain:  # or self.opt.use_encoded_image:
            B_path = replace_last_occurence(A_path, 'train_A', 'train_B')
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)[:1]

        input_dict = {'label': torch.cat((A_tensor, D_tensor, E_tensor)),
                      'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.dataset_size) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
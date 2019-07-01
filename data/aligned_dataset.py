### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### input C (edge images)
        dir_C = '_C'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))

        dir_D = '_D'
        self.dir_D = os.path.join(opt.dataroot, opt.phase + dir_D)
        self.D_paths = sorted(make_dataset(self.dir_D))

        dir_E = '_E'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))[:1]

        B_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)[:1]


        ### input C (edge images)
        C_path, D_path, E_path = self.C_paths[index], self.D_paths[index], self.E_paths[index]
        C, D, E = [Image.open(x) for x in (C_path, D_path, E_path)]
        params = get_params(self.opt, C.size)
        transform_C = transform_D = transform_E = get_transform(self.opt, params)
        C_tensor = transform_C(C.convert('RGB'))[:1]
        D_tensor = transform_D(D.convert('RGB'))[:1]
        E_tensor = transform_E(E.convert('RGB'))[:1]

        input_dict = {'label': A_tensor, 'inst': 0, 'image': B_tensor, 'edge': C_tensor,
                      'context_all': D_tensor, 'context_single': E_tensor,
                      'feat': 0, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
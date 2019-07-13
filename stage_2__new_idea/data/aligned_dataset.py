import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        # dir_A = '_A'  # if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, 'train_A')  # , opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        # if opt.isTrain or opt.use_encoded_image:
        # dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, 'train_B')  # , opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        # ### instance maps
        # if not opt.no_instance:
        #     self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
        #     self.inst_paths = sorted(make_dataset(self.dir_inst))
        #
        # ### load precomputed instance-wise encoded features
        # if opt.load_features:
        #     self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
        #     print('----------- loading features from %s ----------' % self.dir_feat)
        #     self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
        kf = KFold(n_splits=opt.num_nets, random_state=42)
        self.kf_indices = list(kf.split(np.arange(self.dataset_size)))[opt.net_idx][0 if opt.phase == 'train' else 1]

      
    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[self.kf_indices[index]]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if True:  # self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))[:self.opt.input_nc]
        # else:
        #     transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        #     A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[self.kf_indices[index]]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)[:self.opt.output_nc]

        # ### if using instance maps
        # if not self.opt.no_instance:
        #     inst_path = self.inst_paths[index]
        #     inst = Image.open(inst_path)
        #     inst_tensor = transform_A(inst)
        #
        #     if self.opt.load_features:
        #         feat_path = self.feat_paths[index]
        #         feat = Image.open(feat_path).convert('RGB')
        #         norm = normalize()
        #         feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.kf_indices) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

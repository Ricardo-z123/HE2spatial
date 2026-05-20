import torch
import os
import numpy as np
import pandas as pd
import scprep as scp
from PIL import Image, ImageFile
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class HER2STStage1Dataset(torch.utils.data.Dataset):
    """
    Per-slide HER2ST dataset for Stage 1 MLM pretraining.
    One sample = one entire slide (variable L = 300-600 spots), so DataLoader uses batch_size=1.

    Two branches share the same train/eval split logic; they differ only in returned fields:
        branch='spot':  gene expression per spot
        branch='image': H&E patches per spot
    """
    def __init__(self, train=True, fold=0, branch='spot'):
        super().__init__()
        assert branch in ('spot', 'image'), f"branch must be 'spot' or 'image', got {branch}"
        self.train = train
        self.branch = branch

        self.cnt_dir = '/root/autodl-tmp/Her2st/data/ST-cnts'
        self.img_dir = '/root/autodl-tmp/Her2st/data/ST-imgs'
        self.pos_dir = '/root/autodl-tmp/Her2st/data/ST-spotfiles'
        self.r = 224 // 2  # half-side of square H&E patch around each spot center

        gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list

        samples = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']
        te_names = [samples[fold]]
        tr_names = sorted(set(samples) - set(te_names))  # sorted for deterministic ordering across runs
        self.names = tr_names if train else te_names
        self.id2name = dict(enumerate(self.names))

        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        self.loc_dict = {i: m[['x', 'y']].values.astype(int) for i, m in self.meta_dict.items()}

        if branch == 'spot':
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m[gene_list].values))
                for i, m in self.meta_dict.items()
            }  # lib-norm + log1p HVG expression, same recipe as baseline
        else:
            print("Loading imgs ...")
            self.img_dict = {i: self.get_img(i) for i in self.names}
            self.center_dict = {
                i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
                for i, m in self.meta_dict.items()
            }
            if train:
                self.transforms = transforms.Compose([
                    transforms.ColorJitter(0.5, 0.5, 0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=180),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.ToTensor()

    def __len__(self):
        """Number of slides in this split (e.g. 5 train, 1 test)."""
        return len(self.names)

    def __getitem__(self, idx):
        """
        Input:
          idx: slide index in the split (0..len-1)
        Output (branch='spot'):
          gene     [L, 785]   FloatTensor   lib-norm + log1p HVG expression per spot
          coords   [L, 2]     LongTensor    integer (x, y) grid coords per spot
          slide_id str                       slide name like 'A5'
        Output (branch='image'):
          patches  [L, 3, 224, 224] FloatTensor   H&E 224x224 patches per spot (augmented in train, ToTensor only in eval)
          coords   [L, 2]           LongTensor    integer (x, y) grid coords per spot
          slide_id str                             slide name
        """
        name = self.id2name[idx]
        coords = torch.from_numpy(self.loc_dict[name]).long()  # integer (x, y) grid coords per spot

        if self.branch == 'spot':
            gene = torch.from_numpy(self.exp_dict[name]).float()
            return {'gene': gene, 'coords': coords, 'slide_id': name}
            # [L,785] gene expression | [L,2] grid coords | str slide name

        img = self.img_dict[name]
        centers = self.center_dict[name]  # [L, 2] integer pixel centers
        patches = []
        for cx, cy in centers:
            patch = img.crop((cx - self.r, cy - self.r, cx + self.r, cy + self.r))  # crop 224x224 around spot center
            patches.append(self.transforms(patch))
        patches = torch.stack(patches, dim=0)  # stack L per-spot tensors into [L, 3, 224, 224]
        return {'patches': patches, 'coords': coords, 'slide_id': name}
        # [L,3,224,224] H&E patches | [L,2] grid coords | str slide name

    def get_meta(self, name):
        """Join HVG counts and spot positions on spot id, mirroring baseline."""
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'))
        return meta

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name # data/her2st/data/ST-imgs/D/D6
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

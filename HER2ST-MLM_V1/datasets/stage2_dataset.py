import torch
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Pre-computed expression matrices, produced once by data/hvg_her2st.py.
# Stage 2 dataset loads these instead of recomputing lib-norm+log1p in __init__.
PREPROCESSED_EXP_DIR = './data/preprocessed_expression_matrices/her2st'


class HER2STStage2Dataset(torch.utils.data.Dataset):
    """
    Per-spot HER2ST dataset for Stage 2 CLIP-style contrastive learning.
    One sample = one spot (H&E patch + HVG expression + coords + slide id).

    Slide-aware batching is provided by an external SlideBatchSampler that uses
    self.cumlen / self.id2name / self.lengths to make every batch come from a
    single slide — contrastive loss must align intra-tissue spots, not cross-tissue.
    """
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super().__init__()
        self.cnt_dir = '/root/autodl-tmp/Her2st/data/ST-cnts'
        self.img_dir = '/root/autodl-tmp/Her2st/data/ST-imgs'
        self.pos_dir = '/root/autodl-tmp/Her2st/data/ST-spotfiles'
        self.r = 224 // 2  # half-side of square H&E patch around each spot center
        gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()                    # ['A1.tsv.gz', 'A2.tsv.gz', ...]
        names = [i[:2] for i in names]  # ['A1', 'A2', 'A3', ...]

        self.train = train

        samples = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']
        te_names = [samples[fold]]  # held-out slide for this fold
        tr_names = sorted(set(samples) - set(te_names))  # sorted for deterministic ordering across runs
        if train:
            names = tr_names
        else:
            names = te_names
        self.names = names

        print("Loading imgs ...")
        self.img_dict = {i: self.get_img(i) for i in names}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in names}

        # Load pre-computed lib-norm + log1p HVG expression matrices (single source of truth with evel.py).
        self.exp_dict = {}
        for i in names:
            npy_path = f"{PREPROCESSED_EXP_DIR}/{i}/preprocessed_matrix.npy"
            if not os.path.exists(npy_path):
                raise FileNotFoundError(
                    f"Preprocessed expression matrix not found: {npy_path}\n"
                    f"Run 'python data/hvg_her2st.py' first to generate it."
                )
            self.exp_dict[i] = np.load(npy_path).T   # [785, N] -> [N, 785] to match meta row order
            assert self.exp_dict[i].shape[0] == len(self.meta_dict[i]), \
                f"Slide {i}: preprocessed_matrix has {self.exp_dict[i].shape[0]} spots " \
                f"but meta has {len(self.meta_dict[i])}. Re-run hvg_her2st.py."

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }  # [L, 2] integer pixel centers, used both for cropping and for inference write-back

        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  # [L, 2] grid coords per slide

        self.lengths = [len(i) for i in self.meta_dict.values()]  # spots-per-slide list, SlideBatchSampler reads this
        self.cumlen = np.cumsum(self.lengths)                     # global-index -> slide boundary lookup
        self.id2name = dict(enumerate(names))                     # slide index in this split -> slide name

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    # One sample = one spot here (Stage 1 instead yields one window of many spots).
    def __getitem__(self, index):
        """
        Input:
          index: global spot index in [0, cumlen[-1]) across all slides in the split
        Output:
          image       [3, 224, 224]   FloatTensor   H&E patch around spot center (augmented in train, ToTensor in eval)
          position    [2]             FloatTensor   integer (x, y) grid coords (used as positional embedding)
          expression  [785]           FloatTensor   lib-norm + log1p HVG expression
          center      [2]             FloatTensor   integer pixel coords (kept for inference write-back)
          slide_id    scalar          LongTensor    slide index inside this split's id2name
        """
        i = 0
        item = {}
        while index >= self.cumlen[i]:  # walk cumlen to find which slide this global index falls in
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]  # subtract prior slides' spots -> local spot index within slide i
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))  # 224x224 H&E patch
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        item["image"] = patch
        item["position"] = loc
        item["expression"] = exp
        item["center"] = torch.Tensor(center)
        item["slide_id"] = torch.tensor(i)
        return item
        # [3,224,224] image | [2] grid pos | [785] gene | [2] pixel center | scalar slide id

    def __len__(self):
        """Total spot count across all slides in this split."""
        return self.cumlen[-1]

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

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

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'

        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name  # data/her2st/data/ST-imgs/D/D6
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

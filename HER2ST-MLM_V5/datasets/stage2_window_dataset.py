import torch
import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from scipy.spatial import cKDTree

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Pre-computed expression matrices, produced once by data/hvg_her2st.py.
PREPROCESSED_EXP_DIR = './data/preprocessed_expression_matrices/her2st'


class HER2STStage2WindowDataset(torch.utils.data.Dataset):
    # Multiplier that keeps slide blocks apart in the global spot id (max spots/slide is ~712).
    SID_STRIDE = 100000

    """
    Window-mode Stage 2 dataset (paired image + expression) for CLIP-style contrastive learning.

    One sample = one KNN window (k = window_size**2 spatial-neighbour spots on ONE slide),
    carrying BOTH modalities so a contrastive pair can be formed per spot. Mirrors Stage 1's
    build_window_samples (datasets/stage1_dataset.py) but returns image + expression together.

    vs HER2STStage2Dataset (per-spot): only the granularity differs — item -> window, __len__ -> #windows.
    """
    def __init__(self, train=True, fold=0, window_size=10, window_stride=2,
                 anchor_all=False):           # eval mode: EVERY spot is an anchor (stride ignored) -> one window per spot
        super().__init__()
        self.cnt_dir = '/root/autodl-tmp/Her2st/data/ST-cnts'
        self.img_dir = '/root/autodl-tmp/Her2st/data/ST-imgs'
        self.pos_dir = '/root/autodl-tmp/Her2st/data/ST-spotfiles'
        self.r = 224 // 2
        self.train = train
        self.window_size = int(window_size)      # 10*10
        self.window_stride = int(window_stride)  # 2
        self.anchor_all = bool(anchor_all)

        gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list

        samples = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']   # 6 折:A4-B3
        te_names = [samples[fold]]      # fold=0 → ['A4']
        tr_names = sorted(set(samples) - set(te_names))  # {'A5','A6','B1','B2','B3'}
        self.names = tr_names if train else te_names
        self.id2name = dict(enumerate(self.names))     # int→slide {0:'A5', 1:'A6', 2:'B1', 3:'B2', 4:'B3'}, only for print.
        self.name2sid = {n: i for i, n in enumerate(self.names)}   # slide -> int, {'A5':0, 'A6':1, 'B1':2, 'B2':3, 'B3':4}

        print("Loading imgs ...")
        self.img_dict = {i: self.get_img(i) for i in self.names}  # {'A5': <PIL.Image 6975x7424>, 'A6': <PIL.Image ...>, ...}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in self.names} # {'A5': DataFrame[332, 15590], 'A6': DataFrame[360, ...], ...}

        # Load pre-computed lib-norm + log1p HVG expression (single source of truth with evel.py).
        self.exp_dict = {}      
        for i in self.names:
            npy_path = f"{PREPROCESSED_EXP_DIR}/{i}/preprocessed_matrix.npy"
            if not os.path.exists(npy_path):
                raise FileNotFoundError(
                    f"Preprocessed expression matrix not found: {npy_path}\n"
                    f"Run 'python data/hvg_her2st.py' first to generate it."
                )
            self.exp_dict[i] = np.load(npy_path).T   # [785, N] -> [N, 785] to match meta row order
            assert self.exp_dict[i].shape[0] == len(self.meta_dict[i]), \
                f"Slide {i}: preprocessed_matrix rows != meta rows. Re-run hvg_her2st.py."

        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
                            for i, m in self.meta_dict.items()}      # [N,2] pixel centers
        self.loc_dict = {i: m[['x', 'y']].values.astype(int)
                         for i, m in self.meta_dict.items()}         # [N,2] grid coords
        
########################################################################################################

        self.samples = self.build_samples()   # flat list of window dicts across all slides
        print(f"Stage 2 window dataset: {len(self.samples)} windows over {self.names}")

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
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Output (one window; M = window_size**2):
          image       [M, 3, 224, 224]   H&E patch per spot (augmented in train)
          expression  [M, 785]           lib-norm + log1p HVG per spot
          position    [M, 2]             (x, y) grid coords per spot (for PE)
          anchor_pos  int                row of THIS window's anchor within the M spots
                                         (eval reads only that row)
          spot_id     [M]                globally unique spot id; lets the loss detect the same spot
                                         appearing at several slots of the flattened batch
        """
        sample = self.samples[idx]
        name = sample['slide']
        rows = sample['row_indices']                                   # [M] row ids into this slide
        coords = torch.from_numpy(self.loc_dict[name][rows]).float()   # [M, 2]
        exp = torch.from_numpy(self.exp_dict[name][rows]).float()      # [M, 785]

        img = self.img_dict[name]
        centers = self.center_dict[name][rows]                         # [M, 2] pixel centers
        patches = []
        for cx, cy in centers:
            patch = img.crop((cx - self.r, cy - self.r, cx + self.r, cy + self.r))  # 224x224 H&E
            patches.append(self.transforms(patch))
        patches = torch.stack(patches, dim=0)                          # [M, 3, 224, 224]

        return {'image': patches, 'expression': exp, 'position': coords,
                'anchor_pos': sample['anchor_pos'],
                'spot_id': torch.from_numpy(sample['spot_id']).long()}           # [M]

        # one KNN window, M = window_size**2 = 100 spots, all from ONE slide:
        #   image      [100,3,224,224] float 0~1 raw H&E pixels (train: ColorJitter+HFlip+Rot±180
        #                              +ToTensor; eval: ToTensor only). ImageNet mean/std and
        #                              Virchow2 run later, in the model -- these are PIXELS.
        #   expression [100,785]  float32  per-spot lib-norm(10000) + log10(1+x) HVG
        #   position   [100,2]    float32  grid (x,y)  ⚠ float here, every consumer re-casts to
        #                                  .long() for PosEmbed (lossless: coords are 2..26)
        #   anchor_pos int        which of the 100 rows is this window's anchor (eval reads only
        #                         that row)
        #   spot_id    [100]      int64  sid*100000 + row, unique across slides in THIS dataset
        #                         instance.
        # after DataLoader batch_size=8 windows:
        #   image [8,100,3,224,224] | expression [8,100,785] | position [8,100,2]
        #   anchor_pos [8] | spot_id [8,100]        -- all tensors, no python str field

    def build_samples(self):
        """KNN windows: each window = anchor spot + (k-1) nearest spots on the grid.

        train (anchor_all=False): anchors subsampled by window_stride -> ~N/stride^2 windows.
        eval  (anchor_all=True):  EVERY spot is an anchor, in row order -> exactly N windows, so
                                  len(dataset) == n_spots and the output order matches meta /
                                  preprocessed_matrix (evel.py splits by datasize and relies on it).

        'anchor_pos' records where the anchor sits inside row_indices: tree.query returns neighbours
        by distance (anchor first), but row_indices is re-sorted by row id, which loses that. eval
        needs it to pick the anchor's embedding out of the M outputs.
        """
        out = []         # to save sef.samples.
        for name in self.names:
            coords = self.loc_dict[name]                # [N, 2] grid coords
            N = len(coords)
            k = self.window_size * self.window_size     # spots per window (= seq len M) 10*10 =100.
            if N < k:
                # Silently skipping a slide would desynchronise datasize vs preprocessed_matrix in
                # evel.py and corrupt every downstream metric without any error, so refuse loudly.
                raise ValueError(
                    f"{name}: only {N} spots < window_size**2 = {k}. Lower window_size; "
                    f"skipping the slide would break the eval/expression row alignment.")
            tree = cKDTree(coords)   ## KDTree to find cloest 100 spots for each anchor.
            
            ## anchor spot selction !! evel / train.
            if self.anchor_all:
                anchor_indices = list(range(N))   # every spot is an anchor, in row order; used only in evel part.
            else:
                stride = self.window_stride
                anchor_indices = [i for i in range(N)
                                  if int(coords[i, 0]) % stride == 0 and int(coords[i, 1]) % stride == 0]
                if len(anchor_indices) == 0:
                    anchor_indices = list(range(0, N, max(stride * stride, 1)))
                anchor_indices = sorted(anchor_indices)
            # A5 : anchor_indices = [1, 3, 5, 6, 8, 10, 12, 27, 29, 31, 33, 35, ... , 309, 311, 313, 314, 315, 317]
            # row   1 → coords [10, 18]     
            # row   3 → coords [10, 20]
            # ... ...

            n_before = len(out)
            for anchor_idx in anchor_indices:
                _, neighbor_indices = tree.query(coords[anchor_idx], k=k)  # k=10*10 = 100
                # neighbor_indices = [1, 2, 319, 0, 15, 320, 14, 318, 29, 3, ... , 326, 20, 96]

                rows = np.array(sorted(neighbor_indices.tolist()), dtype=int)   # [k]
                # rows = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, ... , 324, 325, 326]

                hit = np.flatnonzero(rows == anchor_idx)
                # hit.size is always 1: the anchor is a tree point (distance 0, always returned)
                assert hit.size == 1, f"{name}: anchor {anchor_idx} not uniquely in its own window"

                out.append({
                    'slide': name,          # A5
                    'row_indices': rows,    # array([0,1,2,...,326])      
                    'anchor_pos': int(hit[0]),        # anchor's row within this window
                    # Global spot id: unique across slides, so the flattened [B * M] batch can tell
                    # which slots are the SAME spot (windows overlap -> a spot occupies several).

                    'spot_id': self.name2sid[name] * self.SID_STRIDE + rows,   
                    # spot_id = sid*100000 + rows, 100000 is enough to separate slides since max spots/slide ~712
                })
            print(f"  {name}: {len(out) - n_before} windows"
                  f"{' (anchor_all: one per spot)' if self.anchor_all else ''}")
        return out
    
    ## Inference part : 使用 window stride = 1. 只选取这个 window的anchor spot embedding; drop other 99. 
                # anchor_pos = where the anchor ended up after the sort above.
                # eval (evel.py) runs with anchor_all=True, so window i IS spot i. The model
                # returns all M=100 embeddings per window, but only the anchor's row is the
                # embedding OF spot i -- the other 99 belong to its neighbours and are dropped.
                #   evel.py L62  apos = batch["anchor_pos"]      # [B]
                #   evel.py L64  widx = torch.arange(B)          # [0,1,...,B-1]
                #   evel.py L74  img_emb[widx, apos]             # [B,M,256] -> [B,256]
                # i.e. row anchor position[w] of window w. Without this field eval could not tell which of
                # the 100 outputs is the spot it is actually evaluating.
                # hit = np.flatnonzero(rows == anchor_idx)
                
                ## Inference flow : 
                # img_emb  [B, M, 256]  =  [8, 100, 256]
                # widx     [0, 1, 2, ..., 7]        ← 第几个窗口
                # anchor spot position     [1, 3, 5, ..., 91]       ← 每个窗口取第几行
                #                 ↓
                # img_emb[widx, apos]  →  [8, 256]   ← 8 个 anchor 的 embedding
    
# out = [
#   # ── A5 的 81 个窗口 ── idx 0..80 ──
#   {'slide':       'A5',
#    'row_indices': array([0,1,2,...,326]),        # [100] 这个窗口的 100 个行号
#    'anchor_pos':  37,                            # ★ anchor 排在这 100 个里的第 37 位
#    'spot_id':     array([0,1,2,...,326])},       # ★ [100] 全局唯一 id = sid*100000 + rows
#   ...                                            # idx 80
#   # ── A6 的 93 个 ── idx 81..173 ──  spot_id = 100000 + rows
#   # ── B1 的 81 个 ── idx 174..254 ── spot_id = 200000 + rows
#   # ── B2 的 72 个 ── idx 255..326 ── spot_id = 300000 + rows
#   # ── B3 的 75 个 ── idx 327..401 ── spot_id = 400000 + rows
# ]
# len(out) == 402 == __len__() == DataLoader 索引范围

    def get_meta(self, name):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        return cnt.join(pos.set_index('id'))

    def get_pos(self, name):
        df = pd.read_csv(self.pos_dir + '/' + name + '_selection.tsv', sep='\t')
        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        df['id'] = [f"{x[i]}x{y[i]}" for i in range(len(x))]
        return df

    def get_cnt(self, name):
        return pd.read_csv(self.cnt_dir + '/' + name + '.tsv', sep='\t', index_col=0)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        return Image.open(pre + '/' + fig_name)



## Flow Details:  HER2STStage2WindowDataset.__init__   (train=True, fold=0)
# L47  self.names       = ['A5','A6','B1','B2','B3']      ← 排除测试折 A4
# L48  self.id2name     = {0:'A5', 1:'A6', 2:'B1', 3:'B2', 4:'B3'}
# L49  self.name2sid    = {'A5':0, 'A6':1, 'B1':2, 'B2':3, 'B3':4}   ★ 只有 stage2 有,造 spot_id 用
#
# L52  self.img_dict    = get_img()  ← 5 张 WSI 大图 PIL (A5 是 6975×7424 那种量级)
#                         → {'A5': PIL, 'A6': PIL, ...}
#
# L54  self.meta_dict   = {i: self.get_meta(i) ...}
#         └─ get_meta → get_cnt(读 .tsv) + get_pos(读 _selection.tsv) → join
#         → {'A5': [332,15590], 'A6': [360,...], ...}
#
# L57  self.exp_dict    = 读 preprocessed_matrix.npy → .T
#         → {'A5': [332,785], 'A6': [360,785], ...}
#      L66 assert 只比对行数,不比对 spot 身份 ⚠
#
# L69  self.center_dict = meta 里取 pixel_x/pixel_y  → {'A5': [332,2] 像素坐标}
# L71  self.loc_dict    = meta 里取 x,y              → {'A5': [332,2] 网格坐标}
#
# L74  self.samples     = self.build_samples()       ★ 窗口划分到此定死
#         └─ for name in ['A5','A6','B1','B2','B3']:
#                └─ cKDTree(loc_dict[name]) + anchor 筛选 + tree.query(k=100)
#         → [402 个 dict]   A5:81 A6:93 B1:81 B2:72 B3:75
#         每个 dict = {'slide', 'row_indices'[100], 'anchor_pos', 'spot_id'[100]}
#                                                    ★★ 后两个是 stage2 新增
#
# L77  self.transforms  = train ? (ColorJitter+HFlip+Rot±180+ToTensor) : ToTensor
#
# ────────── __init__ 结束 ──────────
#
# ⚠ 与 stage1_dataset 的关键差异:
#   1. 没有 branch 分支 —— img_dict / exp_dict / center_dict / loc_dict 全都建,
#      因为一个样本必须同时给出图像和表达才能配成对比学习的正样本对。
#   2. build_samples 排在 loc_dict 之后 (L71 → L74),因为它要读 self.loc_dict。
#      stage1 里 build_samples 在分支之前,读的是局部变量 meta。
#   3. transforms 排在最后,且不分 branch。
#
# eval 模式 (train=False, fold=0, anchor_all=True):
#   names = ['A4'] → 343 spots → anchor_indices = range(343) → 343 个窗口
#   len(dataset) == n_spots,顺序 == meta 行序,evel.py 靠这条按 datasize 切分
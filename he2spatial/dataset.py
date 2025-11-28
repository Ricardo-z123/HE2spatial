import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torchvision import transforms
import pdb


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path, is_train=True):
        # image_path is the path of an entire slice of visium h&e stained image (~2.5GB)

        # spatial_pos_csv
        # barcode name
        # detected tissue boolean
        # x spot index
        # y spot index
        # x spot position (px)
        # y spot position (px)

        # expression_mtx
        # feature x spot (alphabetical barcode order)

        # barcode_tsv
        # spot barcodes - alphabetical order
        self.is_train = is_train
        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features
        self.basic_transform = transforms.Compose([     # 修改1
            transforms.ToTensor(),                      # HWC → CHW，且归一化到 [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)

        if self.is_train:
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)

            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)

        # Convert to tensor
        # image = TF.to_tensor(image)

        image = self.basic_transform(image)

        # Normalize using ImageNet mean and std
        # image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]

        ## grid coordinates ##
        grid_row = int(self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 2].values[0]) #
        grid_col = int(self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 3].values[0])  #

        ## pixel coordinate ##
        v1 = int(self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0])
        v2 = int(self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0])

        ## image patch ##
        image = self.whole_image[(v1 - 112):(v1 + 112), (v2 - 112):(v2 + 112)]
        image = self.transform(image)
        # pdb.set_trace()
        item['image'] = image                  # 修该2, image.float color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # cell x features (6419)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1, v2]

        item['positions'] = torch.tensor([grid_col, grid_row],dtype=torch.long)

        return item

    def __len__(self):
        return len(self.barcode_tsv)
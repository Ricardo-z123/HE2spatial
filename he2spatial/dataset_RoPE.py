import os
import cv2
import pandas as pd
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torchvision import transforms

# Config
NEIGHBOR_RADIUS = 3  # neighborhood radius in grid coordinates
MAX_NEIGHBORS = 25  # max sequence length (e.g., 5x5 = 25 spots)


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path, is_train=True):
        self.is_train = is_train
        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)

        # Load reduced expression matrix
        if reduced_mtx_path.endswith('.npy'):
            self.reduced_expression = np.load(reduced_mtx_path).T  # (cells, features)
        else:
            try:
                self.reduced_expression = np.load(reduced_mtx_path).T
            except:
                print(f"Warning: Cannot load {reduced_mtx_path}")
                self.reduced_expression = np.zeros((len(self.barcode_tsv), 100))

        # Build grid coordinate map for fast neighbor lookup
        # spatial_pos_csv: [barcode, detected, x_grid, y_grid, x_pixel, y_pixel]
        self.grid_map = {
            (row[2], row[3]): i
            for i, row in self.spatial_pos_csv.iterrows()
        }
        self.all_grid_coords = set(self.grid_map.keys())

        # Image transforms
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Finished loading all files")

    def augment_image(self, image):
        angle = random.choice([0, 90, 180, -90])
        image = TF.rotate(image, angle)
        image = self.basic_transform(image)
        return image

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]

        # Get center spot info
        center_row = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode].iloc[0]
        grid_row = center_row[2]
        grid_col = center_row[3]
        v1 = center_row[4]  # pixel x
        v2 = center_row[5]  # pixel y

        # Center spot features and position
        center_expression = self.reduced_expression[idx]
        center_position = np.array([grid_row, grid_col])

        # Build neighbor sequence (center spot first)
        neighbor_expressions = [center_expression]
        neighbor_positions = [center_position]

        for i in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            for j in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                if i == 0 and j == 0:
                    continue

                neighbor_coord = (grid_row + i, grid_col + j)

                if neighbor_coord in self.all_grid_coords:
                    neighbor_idx = self.grid_map[neighbor_coord]
                    neighbor_expressions.append(self.reduced_expression[neighbor_idx])
                    neighbor_positions.append(np.array(neighbor_coord))

                    if len(neighbor_expressions) >= MAX_NEIGHBORS:
                        break
            if len(neighbor_expressions) >= MAX_NEIGHBORS:
                break

        # Pad to MAX_NEIGHBORS
        current_N = len(neighbor_expressions)
        pad_len = MAX_NEIGHBORS - current_N

        expression_pad = np.zeros((pad_len, self.reduced_expression.shape[1]), dtype=np.float32)
        position_pad = np.zeros((pad_len, 2), dtype=np.int64)

        final_expressions = np.vstack(neighbor_expressions + [expression_pad])
        final_positions = np.vstack(neighbor_positions + [position_pad])

        item["reduced_expression"] = torch.tensor(final_expressions, dtype=torch.float32)  # (MAX_N, D)
        item["positions"] = torch.tensor(final_positions, dtype=torch.long)  # (MAX_N, 2)
        item["seq_len"] = torch.tensor(current_N, dtype=torch.long)

        # Image patch
        image = self.whole_image[(v1 - 112):(v1 + 112), (v2 - 112):(v2 + 112), :]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.is_train:
            image = self.augment_image(image)
        else:
            image = self.basic_transform(image)

        item["image"] = image

        return item

    def __len__(self):
        return len(self.barcode_tsv)
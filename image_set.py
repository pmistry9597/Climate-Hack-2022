import pickle
import torch
import numpy as np
from copy import deepcopy
from random import randrange
from datetime import timedelta
from chunkified_npset import ChunkifiedDataset

class ImageSet(torch.utils.data.Dataset):
    def __init__(self, np_set, gen_tables):
        self.in_width = 128
        self.out_width = 64
        self.inner_offset = int((128 - 64) / 2)
        self.np_set = np_set
        self._gen_tables(gen_tables)

    def _gen_tables(self, gen_tables):
        idx_2_time = None
        with open('uk_data_np/idx_2_time', 'rb') as i2t_f:
            idx_2_time = pickle.load(i2t_f)
        i2t_f.close()

        self.corner_and_idxs = gen_tables(idx_2_time)

    def __len__(self):
        return len(self.corner_and_idxs)

    def _crop(self, section, corner, width):
        return section[corner[0]:corner[0]+width, corner[1]:corner[1]+width]
    
    def _in_crop(self, section, corner):
        return self._crop(section, corner, self.in_width)

    # def _out_crop(self, section, corner):
    #     corner[0] += self.inner_offset
    #     corner[1] += self.inner_offset

    #     return self._crop(section, corner, self.out_width)
    
    def __getitem__(self, idx):
        corner, np_idx = self.corner_and_idxs[idx]
        corner = deepcopy(corner)
        in_section = self.np_set[np_idx]
        in_section = self._in_crop(in_section, corner)
        in_section = in_section.astype(np.float32)

        out_section = in_section
        # out_section = self._out_crop(out_section, corner)
        # out_section = out_section.astype(np.float32)

        return (in_section, out_section)
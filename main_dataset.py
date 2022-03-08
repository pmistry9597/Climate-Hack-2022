import pickle
import torch
import numpy as np
from copy import deepcopy
from random import randrange
from datetime import timedelta
from chunkified_npset import ChunkifiedDataset

class CHMainDataset(torch.utils.data.Dataset):
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
        return section[:, corner[0]:corner[0]+width, corner[1]:corner[1]+width]
    
    def _in_crop(self, section, corner):
        return self._crop(section, corner, self.in_width)

    def _out_crop(self, section, corner):
        corner[0] += self.inner_offset
        corner[1] += self.inner_offset

        return self._crop(section, corner, self.out_width)
    
    def __getitem__(self, idx):
        corner, np_idx = self.corner_and_idxs[idx]
        corner = deepcopy(corner)
        in_section = self.np_set[np_idx : np_idx+12]
        in_section = self._in_crop(in_section, corner)
        in_section = in_section.astype(np.float32)

        out_section = self.np_set[np_idx+12 : np_idx+36]
        out_section = self._out_crop(out_section, corner)
        out_section = out_section.astype(np.float32)

        return (in_section, out_section)


class SingleCropGenerator:
    def __init__(self, raw_w, raw_h):
        self.period = 36
        self.crop_width = 128
        self.raw_w = raw_w
        self.raw_h = raw_h

    def _generate_corner(self):
        corner = [randrange(0, self.raw_w - self.crop_width), randrange(0, self.raw_h - self.crop_width)]
        return corner

    def __call__(self, idx_2_time):
        table = []
        for i in range(len(idx_2_time)):
            end_bound = i + self.period
            if end_bound >= len(idx_2_time): #no more 3 hour intervals to check!
                break
            
            if idx_2_time[i] + timedelta(hours=3) != idx_2_time[end_bound]: #if period # of points down the line isn't 3 hours exactly in the future
                continue

            table.append((self._generate_corner(), i))
        return table
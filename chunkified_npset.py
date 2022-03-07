import numpy as np
import os

class ChunkifiedDataset:
    def __init__(self):
        self.arrays_in_chunk = 500
        self.chunk_loaded_start = -1
        self.chunk_loaded_end = -1
        self.data_root = 'uk_data_np'
        self.chunk_root = os.path.join(self.data_root, 'data')
        self._load_filename_data()
        self.chunk_np = None
        self.prev_chunk_np = [None, -1, -1] #prev array, prev bound start, prev bound end

    def _load_filename_data(self):
        all_chunk_names = os.listdir(self.chunk_root)
        self.chunk_nos = set() # stores end of chunk indices that stores names
        self.highest_chunk_no = 0
        self.highest_chunk_name = ''
        for chunk_name in all_chunk_names:
            chunk_name = chunk_name.replace('.npz', '')
            try:
                chunk_no = int(chunk_name)
            except:
                continue
            self.chunk_nos.add(chunk_no)

            if self.highest_chunk_no < chunk_no:
                self.highest_chunk_no = chunk_no
                self.highest_chunk_name = chunk_name
    
    def _get_chunk_bounds(self, idx):
        chunk_start = int(idx / self.arrays_in_chunk) * self.arrays_in_chunk
        chunk_end = chunk_start + self.arrays_in_chunk - 1
        return (chunk_start, chunk_end)
    
    def __len__(self):
        return self.highest_chunk_no + 1

    def _ensure_chunk_loaded(self, bounds):
        end_idx = bounds[1]
        start_idx = bounds[0]
        if self.chunk_loaded_start == start_idx and self.chunk_loaded_end == end_idx:
            return
        if bounds == tuple(self.prev_chunk_np[1:]):
            #do a swap, useful for when slices cross chunk boundaries
            temp = self.chunk_np
            self.chunk_np = self.prev_chunk_np[0]
            self.prev_chunk_np[0] = temp
            
            temp = self.chunk_loaded_start
            self.chunk_loaded_start = self.prev_chunk_np[1]
            self.prev_chunk_np[1] = temp
            
            temp = self.chunk_loaded_end
            self.chunk_loaded_end = self.prev_chunk_np[2]
            self.prev_chunk_np[2] = temp
            return
    
        path = ''
        if end_idx in self.chunk_nos:
            path = os.path.join(self.chunk_root, f'{end_idx}.npz')
        else: #use last chunk if the generated chunk index doesn't exist
            path = self.highest_chunk_name

        self.prev_chunk_np[0] = self.chunk_np
        self.prev_chunk_np[1] = self.chunk_loaded_start
        self.prev_chunk_np[2] = self.chunk_loaded_end
        
        self.chunk_np = np.load(path)['data']
        self.chunk_loaded_start = start_idx
        self.chunk_loaded_end = end_idx

    def _accessslice_within_chunk(self, key: slice, bounds):
        start_bound = bounds[0]
        self._ensure_chunk_loaded(bounds)
        return self.chunk_np[key.start - start_bound: key.stop - start_bound: key.step]

    def __getitem__(self, key):
        if isinstance(key, int):
            bounds = self._get_chunk_bounds(key)
            start_idx, _ = bounds
            self._ensure_chunk_loaded(bounds)
            return self.chunk_np[key - start_idx]

        if isinstance(key, slice):
            start_bounds = self._get_chunk_bounds(key.start)
            end_bounds = self._get_chunk_bounds(key.stop)
            if start_bounds == end_bounds:
                return self._accessslice_within_chunk(key, start_bounds)
            
            if start_bounds[1] + 1 != end_bounds[0]:
                raise KeyError('Bro, your damn slice is massive yo', key)
            
            slice_0 = slice(key.start, start_bounds[1], key.step)
            section_0 = self._accessslice_within_chunk(slice_0, start_bounds)
            slice_1 = slice(end_bounds[0], key.stop, key.step)
            section_1 = self._accessslice_within_chunk(slice_1, end_bounds)
            section = np.concatenate([section_0, section_1], axis=0)
            return section

        raise KeyError('Bro like idek what to do here ok?', key)
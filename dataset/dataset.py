import os
import pathlib
import torch
from torch.utils.data import Dataset
from utils import utils

class VideoVSLDataset(Dataset):
    def __init__(self, 
                 data_path,
                 transform = None):
        self.transform = transform
        self.video_path = utils.get_data_list(data_path)
        self.class_name,self.class_to_idx = utils.get_class(data_path)
        
    def load_container(self,
                   index):
        path = self.video_path[index]
        return av.open(path)
    
    def __getitem__(self,
                    idx):
        container = self.load_container(idx)
        
        indices = utils.sample_frame_indices(clip_len=32, 
                                       frame_sample_rate=1, 
                                       seg_len=container.streams.video[0].frames)
        video = utils.read_video_pyav(container=container, indices=indices)
        class_name = self.video_path[idx].parent.name
        class_name_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(list(video), return_tensors="pt"),class_name_idx
    
    def __len__(self):
        return len(self.video_path) 
    
def main():
    pass 


if __name__ == "__main__":
    print("hello world")
import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from nerv.utils import load_obj, strip_suffix, read_img, VideoReader

from .utils import compact, BaseTransforms, anno2mask, masks_to_boxes_pad


class AtariDataset(Dataset):
    """Dataset for loading Atari dataset."""

    def __init__(
        self,
        data_root,
        atari_transforms,
        split='train',
        max_n_objects=6,
        video_len=20,
        n_sample_frames=6,
        warmup_len=5,
        frame_offset=None,
        load_mask=False,
        filter_enter=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert self.split in ['train', 'val']
        if self.split == 'train':
            self.files = np.load(os.path.join(data_root, 'boxing_train.npy')).astype(np.uint8)
        else:
            self.files = np.load(os.path.join(data_root, 'boxing_val.npy')).astype(np.uint8)

        self.atari_transforms = atari_transforms
        self.max_n_objects = max_n_objects
        self.video_len = video_len
        self.n_sample_frames = n_sample_frames
        self.warmup_len = warmup_len
        self.frame_offset = video_len // n_sample_frames if \
            frame_offset is None else frame_offset
        self.load_mask = load_mask
        self.filter_enter = filter_enter

        self.sample_from_start = True

        self.atari_transforms.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
        ])

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        episode = self.files[idx]

        if self.sample_from_start:
            start = random.randint(0, len(episode) - self.n_sample_frames)
            stop = start + self.n_sample_frames
        else:
            stop = random.randint(1, len(episode))
            start = stop - self.n_sample_frames
        
        segmented_episode = episode[start:stop]
        segmented_episode = [
            self.atari_transforms(Image.fromarray(frame.transpose(1, 2, 0)).convert('RGB'))
            for frame in segmented_episode
        ]

        return {
            'img': torch.stack(segmented_episode, dim=0).float(),
            'error_flag': False,
            'data_idx': idx,
        }

    def __len__(self):
        return len(self.files)
    
    def get_video(self, idx):
        episode = self.files[idx]

        if self.sample_from_start:
            start = random.randint(0, len(episode) - self.n_sample_frames)
            stop = start + self.n_sample_frames
        else:
            stop = random.randint(1, len(episode))
            start = stop - self.n_sample_frames
        
        segmented_episode = episode[start:stop]
        segmented_episode = [
            self.atari_transforms(Image.fromarray(frame.transpose(1, 2, 0)).convert('RGB'))
            for frame in segmented_episode
        ]

        return {
            'video': torch.stack(segmented_episode, dim=0).float(),
            'error_flag': False,
            'data_idx': idx,
        }


class AtariSlotsDataset(AtariDataset):
    """Dataset for loading Atari videos and pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        atari_transforms,
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=10 + 6,
        warmup_len=5,
        frame_offset=None,
        load_img=False,
        load_mask=False,
        filter_enter=True,
    ):
        self.load_img = load_img
        self.load_mask = load_mask

        super().__init__(
            data_root=data_root,
            atari_transforms=atari_transforms,
            split=split,
            max_n_objects=max_n_objects,
            video_len=video_len,
            n_sample_frames=n_sample_frames,
            warmup_len=warmup_len,
            frame_offset=frame_offset,
            load_mask=load_mask,
            filter_enter=filter_enter,
        )

        # pre-computed slots
        self.video_slots = video_slots

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _read_slots(self, idx):
        """Read video frames slots."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]
        try:
            slots = self.video_slots[os.path.basename(video_path)]  # [T, N, C]
        except KeyError:
            raise ValueError
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - slots: [T, N, C] slots extracted from Ataru video frames
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        try:
            slots = self._read_slots(idx)
            data_dict = {
                'data_idx': idx,
                'slots': slots,
                'error_flag': False,
            }
            if self.load_img:
                data_dict['img'] = self._read_frames(idx)
            if self.load_mask:
                data_dict['mask'], data_dict['pres_mask'], \
                    data_dict['bbox'] = self._read_masks(idx)
        # empty video
        except ValueError:
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
        return data_dict


def build_atari_dataset(params, val_only=False, test_set=False):
    """Build Atari video dataset."""
    args = dict(
        data_root=params.data_root,
        atari_transforms=BaseTransforms(params.resolution),
        split='val',
        max_n_objects=6,
        n_sample_frames=params.n_sample_frames,
        warmup_len=params.input_frames,
        frame_offset=params.frame_offset,
        load_mask=params.get('load_mask', False),
        filter_enter=params.filter_enter,
    )

    val_dataset = AtariDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = AtariDataset(**args)
    return train_dataset, val_dataset


def build_atari_slots_dataset(params, val_only=False):
    """Build Atari video dataset with pre-computed slots."""
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=slots['val'],
        atari_transforms=BaseTransforms(params.resolution),
        split='val',
        max_n_objects=6,
        n_sample_frames=params.n_sample_frames,
        warmup_len=params.input_frames,
        frame_offset=params.frame_offset,
        load_img=params.load_img,
        load_mask=params.get('load_mask', False),
        filter_enter=params.filter_enter,
    )
    val_dataset = AtariSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = AtariSlotsDataset(**args)
    return train_dataset, val_dataset

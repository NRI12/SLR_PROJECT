import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T
import random

class VideoTransforms(nn.Module):
    def __init__(self, 
                 resize: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 horizontal_flip_prob: float = 0.5,
                 color_jitter: bool = True,
                 max_frames: Optional[int] = None,
                 output_format: str = 'CTHW'):
        super().__init__()
        self.resize = resize
        self.normalize = normalize
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter
        self.temporal_crop = False
        self.max_frames = max_frames
        self.output_format = output_format
        
        if normalize:
            self.norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if color_jitter:
            self.color_transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        T, H, W, C = video.shape
        video = video.permute(0, 3, 1, 2)  
        
        if self.resize:
            video = torch.nn.functional.interpolate(video, size=self.resize, mode='bilinear', align_corners=False)
        
        video = video.float() / 255.0
        
        if self.training and self.color_jitter:
            video = torch.stack([self.color_transform(frame) for frame in video])
        
        if self.training and random.random() < self.horizontal_flip_prob:
            video = torch.flip(video, dims=[3])
        
        if self.normalize:
            video = torch.stack([self.norm_transform(frame) for frame in video])
        
        if self.output_format == 'CTHW':
            video = video.permute(1, 0, 2, 3)
        elif self.output_format == 'TCHW':
            pass
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        
        return video


class KeypointTransforms:
    def __init__(self,
                 target_frames: int,
                 apply_augment: bool = True,
                 mirror_prob: float = 0.5):
        self.target_frames = target_frames
        self.apply_augment = apply_augment
        self.mirror_prob = mirror_prob

    # data_numpy: [C, T, V, M]
    def __call__(self, data_numpy: np.ndarray) -> np.ndarray:
        data_numpy = self._temporal_pad_or_crop(data_numpy, self.target_frames, random_crop=self.apply_augment)
        if self.apply_augment:
            data_numpy = self.random_choose(data_numpy, size=self.target_frames, random_sample=True)
            data_numpy = self.random_shift(data_numpy)
            data_numpy = self.random_move(data_numpy)
            data_numpy = self.random_mirror(data_numpy, prob=self.mirror_prob)
        # zero-mean per sample on x,y channels
        data_numpy = data_numpy.astype(np.float32)
        mean_xy = data_numpy[:2].reshape(2, -1).mean(axis=1, keepdims=True)
        data_numpy[:2] = (data_numpy[:2].reshape(2, -1) - mean_xy).reshape(2, data_numpy.shape[1], data_numpy.shape[2], data_numpy.shape[3])
        return data_numpy

    def _temporal_pad_or_crop(self, data_numpy: np.ndarray, size: int, random_crop: bool) -> np.ndarray:
        C, T, V, M = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_crop else 0
            data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
            data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
            return data_numpy_paded
        elif T > size:
            begin = random.randint(0, T - size) if random_crop else (T - size) // 2
            return data_numpy[:, begin:begin + size, :, :]
        else:
            return data_numpy

    def random_choose(self, data_numpy, size, random_sample=True):
        C, T, V, M = data_numpy.shape
        if T == size:
            return data_numpy
        elif T < size:
            return self._temporal_pad_or_crop(data_numpy, size, random_crop=random_sample)
        else:
            begin = random.randint(0, T - size)
            return data_numpy[:, begin:begin + size, :, :]

    def random_shift(self, data_numpy):
        C, T, V, M = data_numpy.shape
        data_shift = np.zeros_like(data_numpy)
        valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()
        size = end - begin
        if size <= 0 or size > T:
            return data_numpy
        bias = random.randint(0, T - size)
        data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]
        return data_shift

    def random_move(self, data_numpy,
                    angle_candidate=[-10., -5., 0., 5., 10.],
                    scale_candidate=[0.9, 1.0, 1.1],
                    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    move_time_candidate=[1]):
        C, T, V, M = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])

        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
        return data_numpy

    def random_mirror(self, data_numpy, prob: float = 0.5):
        if np.random.rand() < prob:
            data_numpy = data_numpy.copy()
            data_numpy[0] = -data_numpy[0]
        return data_numpy

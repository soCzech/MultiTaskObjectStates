import os
import glob
import time
import torch
import ffmpeg
import random
import itertools
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import TypeVar, Optional, Iterator

import imgaug as ia
import imgaug.augmenters as iaa


def extract_frames(video_path, fps, size=None, crop=None, start=None, duration=None):
    if start is not None:
        cmd = ffmpeg.input(video_path, ss=start, t=duration)
    else:
        cmd = ffmpeg.input(video_path)

    if size is None:
        info = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"][0]
        size = (info["width"], info["height"])
    elif isinstance(size, int):
        size = (size, size)

    if fps is not None:
        cmd = cmd.filter('fps', fps=fps)
    cmd = cmd.filter('scale', size[0], size[1])

    if crop is not None:
        cmd = cmd.filter('crop', f'in_w-{crop[0]}', f'in_h-{crop[1]}')
        size = (size[0] - crop[0], size[1] - crop[1])

    for i in range(5):
        try:
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            break
        except Exception as e:
            time.sleep(random.random() * 5.)
            if i < 4:
                continue
            print(f"W: FFMPEG file {video_path} read failed!", flush=True)
            if isinstance(e, ffmpeg.Error):
                print("STDOUT:", e.stdout, flush=True)
                print("STDERR:", e.stderr, flush=True)
            raise

    video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
    return video


class ChangeItVideoDataset(Dataset):

    def __init__(self,
                 video_roots,
                 annotation_root=None,
                 file_mode="unannotated",  # "unannotated", "annotated", "all"
                 noise_adapt_weight_root=None,
                 noise_adapt_weight_threshold_file=None,
                 augment=False):

        self.classes = {x: i for i, x in enumerate(sorted(set([os.path.basename(fn) for fn in itertools.chain(*[
            glob.glob(os.path.join(root, "*")) for root in video_roots
        ]) if os.path.isdir(fn)])))}

        self.files = {key: sorted(itertools.chain(*[
            glob.glob(os.path.join(root, key, "*.mp4")) + glob.glob(os.path.join(root, key, "*.webm")) for root in
            video_roots
        ])) for key in self.classes.keys()}

        self.annotations = {key: {
            os.path.basename(fn).split(".")[0]: np.uint8(
                [int(line.strip().split(",")[1]) for line in open(fn).readlines()])
            for fn in glob.glob(os.path.join(annotation_root, key, "*.csv"))
        } for key in self.classes.keys()} if annotation_root is not None else None

        if file_mode == "unannotated":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] in self.annotations[key]:
                        self.files[key].remove(fn)
        elif file_mode == "annotated":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] not in self.annotations[key]:
                        self.files[key].remove(fn)
        elif file_mode == "all":
            pass
        else:
            raise NotImplementedError()

        self.flattened_files = []
        for key in self.classes.keys():
            self.flattened_files.extend([(key, fn) for fn in self.files[key]])

        self.augment = augment

        # Noise adaptive weighting
        if noise_adapt_weight_root is None:
            return

        self.noise_adapt_weight = {}
        for key in self.classes.keys():
            with open(os.path.join(noise_adapt_weight_root, f"{key}.csv"), "r") as f:
                for line in f.readlines():
                    vid_id, score = line.strip().split(",")
                    self.noise_adapt_weight[f"{key}/{vid_id}"] = float(score)

        self.noise_adapt_weight_thr = {line.split(",")[0]: float(line.split(",")[2].strip())
                                       for line in open(noise_adapt_weight_threshold_file, "r").readlines()[1:]}

    def __getitem__(self, idx):
        class_name, video_fn = self.flattened_files[idx]
        file_id = os.path.basename(video_fn).split(".")[0]

        video_frames = extract_frames(video_fn, fps=1, size=(398, 224)).copy()

        if self.augment:
            video_frames = ChangeItVideoDataset.augment_fc(video_frames)
            video_frames = video_frames[:, :, random.randint(0, 398 - 224 - 1):][:, :, :224]
        else:
            video_frames = video_frames[:, :, (398 - 224) // 2:][:, :, :224]
        video_frames = torch.from_numpy(video_frames.copy())

        annotation = self.annotations[class_name][file_id] \
            if self.annotations is not None and file_id in self.annotations[class_name] else None
        video_level_score = self.noise_adapt_weight[f"{class_name}/{file_id}"] - self.noise_adapt_weight_thr[class_name] \
            if hasattr(self, "noise_adapt_weight") else None

        return class_name + "/" + file_id, self.classes[class_name], video_frames, annotation, video_level_score

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.flattened_files)

    def __repr__(self):
        string = f"ChangeItVideoDataset(n_classes: {self.n_classes}, n_samples: {self.__len__()}, " \
                 f"augment: {self.augment})"
        for key in sorted(self.classes.keys()):
            string += f"\n> {key:20} {len(self.files[key]):4d}"
            if hasattr(self, "noise_adapt_weight_thr"):
                len_ = len([
                    fn for fn in self.files[key]
                    if self.noise_adapt_weight[f"{key}/{os.path.basename(fn).split('.')[0]}"] >
                       self.noise_adapt_weight_thr[key]
                ])
                string += f" (above threshold {self.noise_adapt_weight_thr[key]:.3f}: {len_:4d})"
        return string

    @staticmethod
    def augment_fc(video_frames):
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.1),  # vertically flip 10% of all images
                # crop images by -5% to 10% of their height/width
                iaa.Sometimes(0.5, iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                iaa.Sometimes(0.5, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-15, 15),  # rotate by -15 to +15 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 4 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 4), [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 3)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)),
                        # blur image using local medians with kernel sizes between 3 and 5
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                    iaa.Add((-10, 10), per_channel=0.5),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                ], random_order=True)
            ],
            random_order=True
        )
        seq_det = seq.to_deterministic()

        video_frames_augmented = np.empty_like(video_frames)
        for i in range(len(video_frames)):
            video_frames_augmented[i] = seq_det.augment_image(video_frames[i])
        return video_frames_augmented


def identity_collate(items):
    return items


T_co = TypeVar('T_co', covariant=True)


class DistributedDropFreeSampler:

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = len(self.dataset) // self.num_replicas  # type: ignore[arg-type]
        if self.num_samples * self.num_replicas < len(self.dataset):  # type: ignore[arg-type]
            if self.rank < len(self.dataset) % self.num_replicas:  # type: ignore[arg-type]
                self.num_samples += 1
        self.total_size = len(self.dataset)  # type: ignore[arg-type]
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

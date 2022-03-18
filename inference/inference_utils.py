import logging
import os
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from datasets.transforms import ReidTransforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import is_image_file

import datetime
import struct 
from torchsummary import summary

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_all_images(path: Union[str, List[str]]) -> List[str]:
    print(path, len(os.listdir(path)))
    if os.path.isdir(path):
        images = os.listdir(path)
        images = [os.path.join(path, item) for item in images if is_image_file(item)]
        return images
    elif is_image_file(path):
        return [path]
    else:
        raise Exception(
            f"{path} is neither a path to a valid image file nor a path to folder containing images"
        )


class ImageFolderWithPaths(ImageFolder):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class ImageDataset(Dataset):
    def __init__(self, dataset: str, transform=None, loader=pil_loader):
        self.dataset = get_all_images(dataset)
        print(self.dataset)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return (
            img,
            "",
            img_path,
        )  ## Hack to be consistent with ImageFolderWithPaths dataset


def make_inference_data_loader(cfg, path, dataset_class):
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    val_set = dataset_class(path, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
    )
    return val_loader


def create_folders():
    if not os.path.exists('debug'):
        os.makedirs('debug')
    if not os.path.exists('layers'):
        os.makedirs('layers')

def bin_write(f, data):
    data =data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def hook(module, input, output):
    setattr(module, "_value_hook", output)

def exp_input(input_batch):
    i = input_batch.cpu().data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile("debug/input.bin", format="f")
    print("input: ", i.shape)

def print_wb_output(model, first_different=False):
    f = None
    count = 0
    for n, m in model.named_modules():
        print(n)
        if count == 0 and first_different:
            in_output = m._value_hook[1]
        else:
            in_output = m._value_hook
        count = count +1

        print(in_output)
        o = in_output.cpu().data.numpy()
        o = np.array(o, dtype=np.float32)
        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")
        print('------- ', n, ' ------') 
        print("debug  ",o.shape)
        
        if not(' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            continue
        
        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = "layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].cpu().data.numpy()
            w = np.array(w, dtype=np.float32)
            print ("    weights shape:", np.shape(w))
        
        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            print ("    bias shape:", np.shape(b))
        
        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f,b)
            bin_write(f,s)
            bin_write(f,rm)
            bin_write(f,rv)
            print ("    b shape:", np.shape(b))
            print ("    s shape:", np.shape(s))
            print ("    rm shape:", np.shape(rm))
            print ("    rv shape:", np.shape(rv))

        else:
            bin_write(f,w)
            if b.size > 0:
                bin_write(f,b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None

def _inference(model, batch, normalize_with_bn=True, export=False):
    model.eval()
    with torch.no_grad():
        start = datetime.datetime.now()
        data, _, filename = batch
        print(data.shape)
        if export:
            print('Exporting for tkDNN')
            # create folders debug and layers if do not exist
            create_folders()

            # add output attribute to the layers
            for n, m in model.backbone.named_modules():
                m.register_forward_hook(hook)

        _, global_feat = model.backbone(
            data.cuda() if torch.cuda.is_available() else data
        )
        if normalize_with_bn:
            global_feat = model.bn(global_feat)

        if export:
            # export input bin
            exp_input(data)

            print_wb_output(model.backbone, True)

            with open("resnet18_ctl.txt", 'w') as f:
                for item in list(model.backbone.children()):
                    f.write("%s\n" % item)

            summary(model.backbone, (3, 256, 128))

        end = datetime.datetime.now()
        delta = end - start
        delta_ms = int(delta.total_seconds() * 1000) # milliseconds
        print('inference time: ', delta_ms, ' ms ')
        return global_feat, filename


def run_inference(model, val_loader, cfg, print_freq, export=False):
    embeddings = []
    paths = []

    for pos, x in enumerate(val_loader):
        if pos % print_freq == 0:
            log.info(f"Number of processed images: {pos*cfg.TEST.IMS_PER_BATCH}")
        model.cuda()
        embedding, path = _inference(model, x, export=export)
        for vv, pp in zip(embedding, path):
            paths.append(pp)
            embeddings.append(vv.detach().cpu().numpy())
        if export:
            break

    embeddings = np.array(np.vstack(embeddings))
    paths = np.array(paths)
    return embeddings, paths


def create_pid_path_index(
    paths: List[str], func: Callable[[str], str]
) -> Dict[str, list]:
    paths_pids = [func(item) for item in paths]
    pid2paths_index = {}
    for idx, item in enumerate(paths_pids):
        if item not in pid2paths_index:
            pid2paths_index[item] = [idx]
        else:
            pid2paths_index[item].append(idx)
    return pid2paths_index


def calculate_centroids(embeddings, pid_path_index):
    pids_centroids_inds = []
    centroids = []
    for pid, indices in pid_path_index.items():
        inds = np.array(indices)
        pids_vecs = embeddings[inds]
        length = pids_vecs.shape[0]
        centroid = np.sum(pids_vecs, 0) / length
        pids_centroids_inds.append(pid)
        centroids.append(centroid)
    centroids_arr = np.vstack(np.array(centroids))
    pids_centroids_inds = np.array(pids_centroids_inds, dtype=np.str_)
    return centroids_arr, pids_centroids_inds

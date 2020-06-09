import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['Eyes']


class Eyes(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 3

    def __init__(self, root='/root/mitya/dataset', split='train', mode=None, transform=None,
                 base_size=80, crop_size=140, **kwargs):
        super(Eyes, self).__init__()

        base_size=80
        crop_size=140

        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError(
                "Found 0 images in subfolders of: " + self.root + "\n")
        self.count = 0

    def _class_to_index(self, mask):
        return mask.argmax(axis=2)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB').resize((160, 80))
        print(self.images[index])
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index]).resize((160, 80))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(
            int(self.base_size * 0.8), int(self.base_size * 1.2))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = h - oh if oh < h else 0
            padw = w - ow if ow < w else 0

            start = 0 if random.random() < 0.5 else padw
            top = 0 if random.random() < 0.5 else padh

            border = (start, top, padw - start, padh - top)

            img = ImageOps.expand(img, border=border, fill=0)
            mask = ImageOps.expand(mask, border=border, fill=0)
        
        img = img.crop((0, 0, w, h))
        mask = mask.crop((0, 0, w, h))

        # # random crop crop_size
        # w, h = img.size
        # x1 = random.randint(0, w - crop_size)
        # y1 = random.randint(0, h - crop_size)
        # img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random() / 2))
        # final transform
        # img.save(f'/root/mitya/Lightweight-Segmentation/samples/{self.count}.png')
        # self.count += 1
        # if (self.count > 25):
        #     raise Exception()
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    maskpath = os.path.join(mask_folder, filename)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:',
                              imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        split = 'training' if (split == 'train') else 'test'
        img_folder = os.path.join(folder, split+'/images')
        mask_folder = os.path.join(folder, split+'/gts')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        raise Exception("haven't done yet")
        # assert split == 'trainval'
        # print('trainval set')
        # train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        # train_mask_folder = os.path.join(folder, 'gtFine/train')
        # val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        # val_mask_folder = os.path.join(folder, 'gtFine/val')
        # train_img_paths, train_mask_paths = get_path_pairs(
        #     train_img_folder, train_mask_folder)
        # val_img_paths, val_mask_paths = get_path_pairs(
        #     val_img_folder, val_mask_folder)
        # img_paths = train_img_paths + val_img_paths
        # mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = Eyes()
    img, label = dataset[0]

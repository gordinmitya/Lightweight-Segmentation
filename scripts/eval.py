import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.utils.metric import SegmentationMetric
from light.utils.visualize import get_color_pallete
from light.utils.logger import setup_logger
from light.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
import cv2
import numpy as np

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset('eyes', split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                            aux=args.aux, pretrained=True, pretrained_base=False)
        if args.distributed:
            self.model = self.model.module
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

            if True:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                # mask = get_color_pallete(predict, self.args.dataset)
                image = image.cpu().data.numpy().squeeze(0).transpose((1,2,0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                colors = np.array([[255,0,0], [0,255,0], [0,0,255]])

                res = np.zeros((image.shape[0]*3, image.shape[1], 3))
                inp = ((image + 1) * 127.5).astype('int')
                msk = colors[predict]
                res[0:image.shape[0],:,:] = inp
                res[image.shape[0]:image.shape[0]*2,:,:] = msk
                res[image.shape[0]*2:,:,:] = cv2.addWeighted(inp, 0.5, msk, 0.5, 0)
                cv2.imwrite(f'/root/mitya/Lightweight-Segmentation/results/{i}.png', res)
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = False
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}'.format(args.model, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger(args.model, args.log_dir, get_rank(),
                          filename='{}_{}_log.txt'.format(args.model, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()

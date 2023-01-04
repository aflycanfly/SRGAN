# -- coding: utf-8 --
# @Time : 15/2/2022 下午 6:28
# @Author : wkq
import argparse


class Config:

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
        parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
        parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
        parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
        parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
        parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
        opt = parser.parse_args()
        return opt

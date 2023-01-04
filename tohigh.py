# -- coding: utf-8 --
# @Time : 4/3/2022 下午 10:26
# @Author : wkq
import numpy as np
import glob
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from models import GeneratorResNet, Discriminator
from torchvision.utils import save_image, make_grid
from option import Config
from torch.autograd import Variable

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageTestDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        or_image = self.transform(img)

        return or_image

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    # Initialize generator and discriminator
    generator = GeneratorResNet()
    if cuda:
        generator = generator.cuda()
    generator.load_state_dict(torch.load("./saved_models/generator_1.pth"))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        dataset=ImageTestDataset("./low_images/", hr_shape=(64, 64)),
        batch_size=4,
    )

    for i, imgs_lr in enumerate(dataloader):
        imgs_lr = Variable(imgs_lr.type(Tensor))
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Save image grid with upsampled inputs and SRGAN outputs
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, gen_hr), -1)
        save_image(img_grid, "high_images/%d.png" % i, normalize=False)


import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import os
import random

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1, shuffle=False):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=shuffle)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    modified_split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        modified_scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
            modified_scores.append(-entropy(pyx))
        split_scores.append(np.exp(np.mean(scores)))
        modified_split_scores.append(np.exp(np.mean(modified_scores)))

    return np.mean(split_scores), np.mean(modified_split_scores)

if __name__ == '__main__':
    class ImageNet(torch.utils.data.Dataset):
        def __init__(self, image_dir, transform):
            self.image_dir = image_dir
            self.transform = transform
            self.dataset = []
            self.preprocess()
            print(len(self.dataset))

        def preprocess(self):
            for name in os.listdir(self.image_dir):
                img = Image.open(os.path.join(self.image_dir, name))
                if (img.mode != 'RGB'):
                    continue
                self.dataset.append(name)
            random.shuffle(self.dataset)


        def __getitem__(self, index):
            filename = self.dataset[index]
            image = Image.open(os.path.join(self.image_dir, filename))
            return self.transform(image)

        def __len__(self):
            return len(self.dataset)

    image_dir = './test/'
    transform = []
    transform.append(transforms.CenterCrop(256))
    transform.append(transforms.Resize((256, 256), Image.BILINEAR))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    dataset=ImageNet(image_dir, transform)

    print ("Calculating Inception Score...")
    print (inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=1, shuffle=False))

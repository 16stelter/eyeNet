import torch
from torch import nn, optim, conv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import dataset
import numpy as np


class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()
        self.model = fcn_resnet50(pretrained_backbone=True, num_classes=2, progress=True)
        self.opt = optim.SGD(self.model.parameters(), lr=0.1)

        g = self.gaussian(25, 11.2)
        kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
        self.model.register_parameter(name='kernel', param=torch.nn.Parameter(kernel, requires_grad=False))

        # center bias
        self.center_bias = torch.from_numpy(np.load("../cv2_project_data/center_bias_density.npy"))
        self.register_parameter(name='density', param=torch.nn.Parameter(torch.log(self.center_bias), requires_grad=False))

    def forward(self, x):
        pred = self.model(x)
        smooth = conv2d(pred["out"][:, 1, :, :].unsqueeze(1), self.model.get_parameter('kernel').unsqueeze(0).unsqueeze(0), padding='same')
        # class 1 is activation, 0 is background
        return torch.add(smooth, self.center_bias)

    def train_loop(self, epochs, ds_loader):
        self.model.train()
        for e in range(epochs):
            print("Starting Epoch", e)
            for step, (x, y) in enumerate(tqdm(ds_loader)):
                pred = self.forward(x)
                loss = F.binary_cross_entropy_with_logits(pred, y)
                print(loss)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            torch.save(self.model.state_dict(), '../checkpoints/{}.pth'.format(e))
        #img = self.forward(x).detach().numpy()[0][0]
        #plt.imshow(img, interpolation='nearest')
        #plt.show()

    def gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        device, dtype = None, None
        if isinstance(sigma, torch.Tensor):
            device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        return gauss / gauss.sum()


if __name__ == "__main__":
    ds = dataset.FixationDataset('../cv2_project_data',
                                 '../cv2_project_data/train_images.txt',
                                 '../cv2_project_data/train_fixations.txt',
                                 Compose([ToTensor()]),
                                 Compose([ToTensor()])
                                 )
    ds_loader = DataLoader(ds, batch_size=1, shuffle=True)

    net = EyeNet()

    net.train_loop(10, ds_loader)
import random

import torch
from torch import nn, optim, conv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor, RandomApply, ColorJitter, RandomPosterize, RandomSolarize, \
    RandomAdjustSharpness
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataset
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
EARLY_TERMINATION = 5


class EyeNet(nn.Module):
    def __init__(self, from_checkpoint=None):
        super(EyeNet, self).__init__()
        self.model = fcn_resnet50(pretrained_backbone=True, num_classes=2, progress=True).to(DEVICE)
        self.opt = optim.SGD(self.model.parameters(), lr=0.1)
        self.min_val_loss = np.inf
        self.best_val_epoch = -1

        self.writer = SummaryWriter()

        g = self.gaussian(25, 11.2)
        kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
        self.model.register_parameter(name='kernel', param=torch.nn.Parameter(kernel, requires_grad=False))

        # center bias
        self.center_bias = torch.from_numpy(np.load("../cv2_project_data/center_bias_density.npy")).to(DEVICE)
        self.register_parameter(name='density', param=torch.nn.Parameter(torch.log(self.center_bias), requires_grad=False))

        if from_checkpoint:
            self.load_model(from_checkpoint)

        print("Init successful...")
        print("DEVICE is " + str(DEVICE))

    def forward(self, x):
        pred = self.model(x)
        smooth = conv2d(pred["out"][:, 1, :, :].unsqueeze(1), self.model.get_parameter('kernel').unsqueeze(0).unsqueeze(0), padding='same')
        # class 1 is activation, 0 is background
        return torch.add(smooth, self.center_bias)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))  # map_location=torch.device('cpu')

    def train_loop(self, epochs, ds_loader, vs_loader):
        self.model.train()
        early_term_count = 0
        for e in range(epochs):
            print("Starting Epoch", e)
            # training on train dataset
            self.model.train()
            for step, (x, y) in enumerate(tqdm(ds_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = self.forward(x)
                loss = F.binary_cross_entropy_with_logits(pred, y)
                self.writer.add_scalar("loss/train", loss, e*step)
                #print(loss)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            # evaluation on val dataset
            self.model.eval()
            val_loss = 0.0
            for step, (x, y) in enumerate(tqdm(vs_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = self.forward(x)
                val_loss += F.binary_cross_entropy_with_logits(pred, y).item()
            val_loss = val_loss / len(vs_loader)
            self.writer.add_scalar("loss/val", val_loss, e)
            print("Validation loss was " + str(val_loss) + ".")
            if self.min_val_loss > val_loss:
                print("This is an improvement by " + str(self.min_val_loss - val_loss) + ".")
                self.min_val_loss = val_loss
                self.best_val_epoch = e
                early_term_count = 0
            else:
                early_term_count += 1
                print("No improvement. " + str(early_term_count) + " / " + str(EARLY_TERMINATION))
                if(early_term_count >= EARLY_TERMINATION):
                    break
            torch.save(self.model.state_dict(), '../checkpoints/{}.pth'.format(e))
        print("Training completed. Best validation epoch was checkpoint " + str(self.best_val_epoch) + ".")
        #img = self.forward(x).detach().numpy()[0][0]
        #plt.imshow(img, interpolation='nearest')
        #plt.show()

    def gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        device, dtype = DEVICE, None
        if isinstance(sigma, torch.Tensor):
            device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def predict_to_file(self, data):
        for step, (x, y) in enumerate(tqdm(vsloader)):
            img = self.forward(x).detach().numpy()[0][0]
            plt.imshow(img, interpolation='nearest')
            plt.savefig("./{}.png".format(step))
            plt.imshow(y[0][0], interpolation='nearest')
            plt.savefig("./{}_t.png".format(step))


if __name__ == "__main__":
    transforms = Compose([ToTensor(),
                          RandomApply(torch.nn.ModuleList([ColorJitter(0.1, 0.1, 0.1, 0.1)]), 0.1),
                          RandomSolarize(0.5, 0.1),
                          RandomAdjustSharpness(random.uniform(0.5, 1.5), 0.1)])

    ds = dataset.FixationDataset('../cv2_project_data',
                                 '../cv2_project_data/train_images.txt',
                                 '../cv2_project_data/train_fixations.txt',
                                 transforms,
                                 Compose([ToTensor()])
                                 )
    dsloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    vs = dataset.FixationDataset('../cv2_project_data',
                                 '../cv2_project_data/val_images.txt',
                                 '../cv2_project_data/val_fixations.txt',
                                 Compose([ToTensor()]),
                                 Compose([ToTensor()])
                                 )
    vsloader = DataLoader(vs, batch_size=BATCH_SIZE, shuffle=True)

    net = EyeNet()

    net.train_loop(EPOCHS, dsloader, vsloader)

    # tsloader = DataLoader(vs, batch_size=BATCH_SIZE, shuffle=False) # TODO: This is currently the validation file!!!!
    # net.predict_to_file(tsloader)


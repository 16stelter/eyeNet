import torch
from torch import nn, optim, conv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import dataset
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10


class EyeNet(nn.Module):
    def __init__(self, from_checkpoint=None):
        super(EyeNet, self).__init__()
        self.model = fcn_resnet50(pretrained_backbone=True, num_classes=2, progress=True)
        if from_checkpoint:
            self.load_model(from_checkpoint)
        self.opt = optim.SGD(self.model.parameters(), lr=0.1)
        self.min_val_loss = np.inf
        self.best_val_epoch = -1

        g = self.gaussian(25, 11.2)
        kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
        self.model.register_parameter(name='kernel', param=torch.nn.Parameter(kernel, requires_grad=False))

        # center bias
        self.center_bias = torch.from_numpy(np.load("../cv2_project_data/center_bias_density.npy"))
        self.register_parameter(name='density', param=torch.nn.Parameter(torch.log(self.center_bias), requires_grad=False))

        print("Init successful...")
        print("DEVICE is " + str(DEVICE))

    def forward(self, x):
        pred = self.model(x)
        smooth = conv2d(pred["out"][:, 1, :, :].unsqueeze(1), self.model.get_parameter('kernel').unsqueeze(0).unsqueeze(0), padding='same')
        # class 1 is activation, 0 is background
        return torch.add(smooth, self.center_bias)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train_loop(self, epochs, ds_loader, vs_loader):
        self.model.train()
        for e in range(epochs):
            print("Starting Epoch", e)
            # training on train dataset
            self.model.train()
            for step, (x, y) in enumerate(tqdm(ds_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = self.forward(x)
                loss = F.binary_cross_entropy_with_logits(pred, y)
                print(loss)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            # evaluation on val dataset
            self.model.eval()
            val_loss = 0.0
            steps = 0
            for step, (x, y) in enumerate(tqdm(vs_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = self.forward(x)
                val_loss += F.binary_cross_entropy_with_logits(pred, y)
                steps += 1
            val_loss = val_loss / steps
            print("Validation loss was " + str(val_loss)+ ".")
            if self.min_val_loss > val_loss:
                self.min_val_loss = val_loss
                self.best_val_epoch = e
                print("This is an improvement by " + str(self.min_val_loss - val_loss) + ".")
            torch.save(self.model.state_dict(), '../checkpoints/{}.pth'.format(e))
        print("Training completed. Best validation epoch was checkpoint " + str(self.best_val_epoch) + ".")
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
    dsloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    vs = dataset.FixationDataset('../cv2_project_data',
                                 '../cv2_project_data/val_images.txt',
                                 '../cv2_project_data/val_fixations.txt',
                                 Compose([ToTensor()]),
                                 Compose([ToTensor()])
                                 )
    vsloader = DataLoader(vs, batch_size=BATCH_SIZE, shuffle=True)

    net = EyeNet()
    net = net.to(DEVICE)

    net.train_loop(EPOCHS, dsloader, vsloader)

import torch
import torch.nn as nn
import torch.nn.functional as F
NOISE_DIM = 10

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(MyModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim + 2 + 3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 4096)
        
        self.conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)#-->128 * 8 * 8
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)#--> 64 * 16 * 16
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)#--> 32 * 32 * 32
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 1, 1, stride=1, padding=1)#--> 1 * 30 * 30

    def forward(self, z, ParticleMomentum_ParticlePoint):
        x = F.leaky_relu(self.fc1(torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        EnergyDeposit = x.view(-1, 256, 4, 4)
        EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit)))
        EnergyDeposit = self.conv5(EnergyDeposit)

        return EnergyDeposit

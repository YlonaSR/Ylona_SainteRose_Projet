
import torch
import torch.nn as nn
import torch.nn.functional as F

#DEBUT DU CODE DE UNET_PARTS

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None): 
        #nbre de canaux d'entrée, 3 pour RGB; nbre de can de sortie apres la double convolution ; nbre de canaux intermediaires
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            self.double_conv = nn.Sequential( #Permet de crer un modele avec plusieurs couches
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), #filtre de 3x3 avec un padding de 1 sans biais 
            nn.BatchNorm2d(mid_channels), #Normalise les valeurs de sortie
            nn.ReLU(inplace=True), #introduit la non-linearité
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), #Double convolution 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True) #derniere activation
        )

    def forward(self, x): #x : donnees d'entrée
        return self.double_conv(x) #Applique les suites de couches definies dans init


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #max pooling avec un facteur de 2, reduction de 50% de la taille spatiale
            DoubleConv(in_channels, out_channels) #2 convolutions sucessives pour extaire des caracteristiques
        )

    def forward(self, x):
        return self.maxpool_conv(x) #max pooling effectue un downscaling plus une double convolution


class Up(nn.Module): #Augmenter la resolution des donnees d'entrée
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True): # bilinear : soit par interpolation bilineaire ou par convolution transposée
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #double la taille de l'image
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) #convolution transposée qui reconstruit les détails
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1) #agrandi la taille, augmentation de la resolution
        # input is CHW 
        #Calcule la differnece des dimensions entre x1 et x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # Ajuster les dimensions de x1
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1) #Concatenation de x1 eet x2 sur la diamesion des canaux 
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__() #
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #Passer de 1 a 3
        # noyau de taille 1x1, chaque pixel est tranforme infedpndamment dans melanges les informations

    def forward(self, x):
        return self.conv(x) # en sortie on a la transformée


#DEBUT DU CODE DE UNET_MODEL



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False): #nbre de canaux d'entrée; nbre de claases de segmentation
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #Phase d'encodage, convolution descendante qui extraient des caracteristiques à plsrs échelles
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        #Phase de décodage, convolution ascendante, qui reconstruit une image de segmentation en combinant les caractéristiques
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        #Couche de sortie, qui reproduits les logits finaux pour la classification des pixels
        self.outc = (OutConv(64, n_classes))

    def forward(self, x): #Propagation
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch.nn.functional as F

# ===== MODÈLE UNET =====

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
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

# ===== DATASET =====

class SimDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + 
                                glob.glob(os.path.join(mask_dir, "*.jpg")))
        self.transform = transform
        
        # Vérifier que nous avons le même nombre d'images et de masques
        assert len(self.img_files) == len(self.mask_files), f"Nombre d'images ({len(self.img_files)}) != nombre de masques ({len(self.mask_files)})"

    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        # Charger l'image et le masque
        img = Image.open(self.img_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')  # Convertir en niveaux de gris
        
        # Redimensionnement
        img = img.resize((224, 224), Image.LANCZOS)
        mask = mask.resize((224, 224), Image.LANCZOS)
        
        # Conversion en tenseurs
        img_tensor = transforms.ToTensor()(img)
        mask_tensor = transforms.ToTensor()(mask)
        
        # Normalisation de l'image
        img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])(img_tensor)
        
        # Binariser le masque (0 ou 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor

# ===== FONCTIONS D'ENTRAÎNEMENT =====

def dice_coefficient(pred, target, smooth=1e-6):
    """Calcule le coefficient de Dice"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(pred, target, smooth=1e-6):
    """Loss fonction basée sur le coefficient de Dice"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combined_loss(pred, target):
    """Combinaison de BCE et Dice loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, device='cpu'):
    """Fonction d'entraînement du modèle"""
    best_model_wts = model.state_dict()
    best_dice = 0.0
    
    # Listes pour stocker les métriques
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Chaque époque a une phase d'entraînement et de validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_dice = 0.0
            
            # Itérer sur les données
            for inputs, masks in tqdm(dataloaders[phase], desc=f'{phase} phase'):
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                # Remettre à zéro les gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    dice = dice_coefficient(outputs, masks)
                    
                    # Backward pass seulement en phase d'entraînement
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistiques
                running_loss += loss.item() * inputs.size(0)
                running_dice += dice.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')
            
            # Stocker les métriques
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_dice_scores.append(epoch_dice)
            else:
                val_losses.append(epoch_loss)
                val_dice_scores.append(epoch_dice)
                
                # Sauvegarder le meilleur modèle
                if epoch_dice > best_dice:
                    best_dice = epoch_dice
                    best_model_wts = model.state_dict()
        
        print()
    
    print(f'Meilleur Dice de validation: {best_dice:.4f}')
    
    # Charger les meilleurs poids du modèle
    model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, train_dice_scores, val_dice_scores

def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualise les prédictions du modèle"""
    model.eval()
    
    # Récupérer un batch de données
    inputs, masks = next(iter(dataloader))
    inputs = inputs.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
    
    # Convertir en numpy pour l'affichage
    inputs_np = inputs.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Dénormaliser les images pour l'affichage
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(min(num_samples, len(inputs))):
        # Dénormaliser l'image
        img = inputs_np[i] * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        # Image originale
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image originale')
        axes[i, 0].axis('off')
        
        # Masque vérité terrain
        axes[i, 1].imshow(masks_np[i, 0], cmap='gray')
        axes[i, 1].set_title('Masque vérité terrain')
        axes[i, 1].axis('off')
        
        # Prédiction
        axes[i, 2].imshow(preds_np[i, 0], cmap='gray')
        axes[i, 2].set_title('Prédiction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ===== SCRIPT PRINCIPAL =====

if __name__ == "__main__":
    # Configuration des chemins
    #img_dir = r"I:/Chercheurs/Lerouge_Sophie/P_Ylona_SteRose_2025/Experiences/Code Python_DeepLearning/Images"
    img_dir = r"I:\Chercheurs\Lerouge_Sophie\P_Ylona_SteRose_2025\Experiences\Analyse_Python_DeepLearning\Images\Images_Augmented_data_Gaussian_Noise"
    mask_dir = r"I:\Chercheurs\Lerouge_Sophie\P_Ylona_SteRose_2025\Experiences\Analyse_Python_DeepLearning\Images\Masques_Augmented_data_Gaussian_Noise"
    #mask_dir = r"I:/Chercheurs/Lerouge_Sophie/P_Ylona_SteRose_2025/Experiences/Code Python_DeepLearning/Masques"
    
    # Vérifier la disponibilité du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Utilisation du device: {device}')
    
    # Créer le dataset
    dataset = SimDataset(img_dir, mask_dir)
    print(f"Nombre total d'images: {len(dataset)}")
    
    # Division train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Images d'entraînement: {len(train_dataset)}")
    print(f"Images de validation: {len(val_dataset)}")
    
    # Créer les dataloaders
    batch_size = 4  # Réduire si problème de mémoire
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    # Créer le modèle
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    
    # Optionnel : Scheduler pour ajuster le learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    print("Début de l'entraînement...")
    
    # Entraîner le modèle
    model, train_losses, val_losses, train_dice, val_dice = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=30, device=device)
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Modèle sauvegardé sous 'unet_model.pth'")
    
    # Visualiser les résultats d'entraînement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Evolution de la Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_dice, label='Train Dice')
    ax2.plot(val_dice, label='Validation Dice')
    ax2.set_title('Evolution du score Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Visualiser quelques prédictions
    visualize_predictions(model, dataloaders['val'], device)
import torch
from torch_Unet import UNet
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import skimage as ski
from skimage import measure


print('Wait...')
# Charger le modèle
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load("C:/Users/p0137947/printability_model.pth", weights_only=True))
#model.load_state_dict(torch.load("C:/Users/p0137947/unet_model.pth", weights_only=True))
model.eval()

# Ouvrir une boîte de dialogue pour choisir un dossier
root = tk.Tk()
root.withdraw()  # Cacher la fenêtre principale
image_files = filedialog.askopenfilenames(title="Sélectionner les images", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])

# Vérifier le dossier sélectionné
if not image_files:
    print("Aucun dossier sélectionné.")
    exit()

print(f"Images sélectionnées : {image_files}")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Appliquer la transformation et traiter les images
for img_path in image_files:
    print(f"Traitement de l'image : {img_path}")
    
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

    with torch.no_grad():
        output = model(image)

    # Convertir la sortie en tableau numpy
    output_image = output.squeeze().cpu().numpy()
    threshold = 0.5
    binary_mask = (output_image > threshold).astype(np.uint8)  # Convertir en 0 (noir) ou 1 (blanc)

    # Superposition des images et des masques
    final_img = cv2.imread(img_path)  # Lire l'image
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  # Conversion en RGB

    # Redimensionner le masque pour qu'il corresponde à la taille de l'image originale
    final_mask = cv2.resize(binary_mask, (final_img.shape[1], final_img.shape[0]))

    print("Préprocessing du masque...")
    mask_median = cv2.medianBlur(final_mask, 3)
    
    # 2. Lissage gaussien léger
    mask_float = mask_median.astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_float, (3, 3), 0.8)
    mask_smooth = (mask_smooth > 0.5).astype(np.uint8)
    
    # 3. Opérations morphologiques finales
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel, iterations=1)

    print("Détection de contours précise...")
    # Utilisation de cv2.findContours avec approximation Douglas-Peucker
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_contours = []
    
    for contour in contours:
        # Approximation Douglas-Peucker pour lisser le contour
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Précision très élevée
        approx = cv2.approxPolyDP(contour, epsilon, True)
        refined_contours.append(approx)

    print("Calculs des mesures...")
    pixel_size = 1.265 #Images de Aude
        
    # Calcul de l'aire
    total_area_pixels = np.sum(mask_final > 0)
    total_area = total_area_pixels * (pixel_size ** 2)
    
    # Calcul du périmètre avec méthode améliorée
    total_perimeter = 0
    
    for contour in contours:
        # Méthode 1: Périmètre direct d'OpenCV
        perimeter_cv = cv2.arcLength(contour, True)
        
        # Méthode 2: Calcul manuel avec distance euclidienne
        contour_points = contour.reshape(-1, 2)
        if len(contour_points) > 2:
            # Calculer la distance entre points consécutifs
            distances = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
            # Ajouter la distance du dernier au premier point (contour fermé)
            last_distance = np.sqrt(np.sum((contour_points[-1] - contour_points[0])**2))
            perimeter_manual = np.sum(distances) + last_distance
            
            # Utiliser la moyenne des deux méthodes pour plus de précision
            perimeter_avg = (perimeter_cv + perimeter_manual) / 2
            total_perimeter += perimeter_avg
    
    total_perimeter *= pixel_size

    #print(f"Aire précise: {total_area:.2f} µm²")
    #print(f"Périmètre précis: {total_perimeter:.2f} µm")

    # Créer la figure avec les sous-graphiques corrects
# Affichage 1: Image originale
    plt.figure("Image originale")
    plt.imshow(final_img)
    plt.title("Image originale")
    plt.axis("off")
    

    """
    # Affichage 2: Masque binaire
    plt.figure("Masque binaire")
    plt.imshow(mask_final, cmap='gray')
    plt.title("Masque binaire")
    plt.axis("off")
    plt.show()

    # Affichage 3: Superposition simple
    final_mask_color = cv2.cvtColor(mask_final * 255, cv2.COLOR_GRAY2RGB)
    alpha = 0.5
    superposition = cv2.addWeighted(final_img, 1-alpha, final_mask_color, alpha, 0)

    plt.figure("Superposition avec masque")
    plt.imshow(superposition)
    plt.title("Superposition avec masque")
    plt.axis("off")
    plt.show()
    """
    
    # Affichage 4: Contours précis
    mask_preprocessed_color = cv2.cvtColor(mask_final * 255, cv2.COLOR_GRAY2RGB)
    superposition_improved = cv2.addWeighted(final_img, 0.7, mask_preprocessed_color, 0.3, 0)

    #plt.figure("Contours précis")
    #plt.imshow(superposition_improved)
    plt.imshow(final_img)
    for contour in refined_contours:
        contour_points = contour.reshape(-1, 2)
        plt.plot(contour_points[:, 0], contour_points[:, 1], color='red', linewidth=1)
    plt.title(f"Contours précis\nAire: {total_area:.1f} µm², Périmètre: {total_perimeter:.1f} µm")
    plt.axis("off")
    plt.show()


    # Ancienne méthode (regionprops) pour comparaison
    #props = measure.regionprops(final_mask.astype(int))
    props = measure.regionprops(mask_final.astype(int))

    if props:
        area = props[0].area * (pixel_size ** 2)
        perim = props[0].perimeter * pixel_size
        print(f"Ancienne méthode - Aire: {area:2f} µm², Périmètre: {perim:.2f} µm")
    
    # Nouvelle méthode
    print(f"Nouvelle méthode - Aire: {total_area:.2f} µm², Périmètre: {total_perimeter:.2f} µm")
    print("-" * 50)

print("Traitement terminé!")

printability_old = (perim**2) / (16*area)
printability_new = (total_perimeter**2) / (16*total_area)
print(f"Printability old method {printability_old:.2f}")
print(f"Printability new method {printability_new:.2f}")

real_area = total_area / 1000000
print(f"Real area: {real_area:.2f}")
theorical_area = 4

theorical_area = int(input("What is the theorical area (in mm²) : "))
print("Theorical area: ", theorical_area)

data = [real_area,theorical_area]
std_dev = np.std(data)
print(f"Standard deviation: {std_dev:.2f}")

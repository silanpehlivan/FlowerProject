import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, canny, corner_harris
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- Sabitler ---
DATASET_PATH = 'flower_photos'
IMAGE_SIZE = (150, 150) # Sabit boyut
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# --- Öznitelik Çıkarma Fonksiyonları ---

def extract_color_features(image_path, target_size=IMAGE_SIZE):
    """
    Görseli yükler, yeniden boyutlandırır, normalize eder ve RGB + HSV kanallarının
    ortalama, standart sapma, min, max değerlerini çıkarır (geliştirilmiş öznitelikler).
    """
    try:
        # Dosya varlığını kontrol et
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")
        
        # 1. Görseli yükle
        img = imread(image_path)
        
        # Resim en az 3 kanallı olmalı (RGB)
        if len(img.shape) < 3 or img.shape[2] < 3:
            raise ValueError("Resim RGB formatında değil veya hatalı formatta")
        
        # 2. Sabit bir boyuta yeniden boyutlandır
        # Anti-aliasing kullanarak daha iyi kalite elde et
        img_resized = resize(img, target_size, anti_aliasing=True)
        
        # RGB kanallarını ayır
        R = img_resized[:, :, 0]
        G = img_resized[:, :, 1]
        B = img_resized[:, :, 2]
        
        # RGB istatistikleri - ortalama, std, min, max
        rgb_features = [
            R.mean(), R.std(), R.min(), R.max(),
            G.mean(), G.std(), G.min(), G.max(),
            B.mean(), B.std(), B.min(), B.max()
        ]
        
        # HSV renk uzayına dönüştür - renge daha duyarlı öznitelikler
        hsv_img = rgb2hsv(img_resized)
        H = hsv_img[:, :, 0]
        S = hsv_img[:, :, 1]
        V = hsv_img[:, :, 2]
        
        # HSV istatistikleri - ortalama, std
        hsv_features = [
            H.mean(), H.std(),
            S.mean(), S.std(),
            V.mean(), V.std()
        ]
        
        # Toplam renk öznitelikleri
        color_features = rgb_features + hsv_features
        
        return color_features
        
    except Exception as e:
        # Hatalı görselleri atlamak için None döndür
        return None

def extract_texture_features(image_path, target_size=IMAGE_SIZE):
    """
    Gri tonlamalı görselden LBP ve GLCM doku özniteliklerini çıkarır.
    (Bu fonksiyon Phase 4'te tamamlanacaktır)
    """
    """
    Gri tonlamalı görselden LBP ve GLCM doku özniteliklerini çıkarır.
    """
    try:
        # Dosya varlığını kontrol et
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")
        
        # 1. Görseli yükle ve yeniden boyutlandır (renk özniteliklerinde yapıldığı gibi)
        img = imread(image_path)
        
        # Resim en az 3 kanallı olmalı (RGB)
        if len(img.shape) < 3 or img.shape[2] < 3:
            raise ValueError("Resim RGB formatında değil veya hatalı formatta")
        
        img_resized = resize(img, target_size, anti_aliasing=True)
        
        # 2. Gri tonlamaya çevir ve 0-255 aralığına ölçekle (GLCM için tam sayı gerekli)
        gray_img = rgb2gray(img_resized)
        # 0-1 aralığındaki float'ı 0-255 aralığındaki uint8'e dönüştür
        # GLCM için seviye sayısını azaltmak (örneğin 64) performansı artırır ve daha stabil sonuçlar verir.
        gray_img_uint8 = (gray_img * 255).astype(np.uint8)
        
        # --- LBP Öznitelikleri ---
        # P=8, R=1, 'uniform' LBP kullan
        radius = 1
        n_points = 8 * radius
        # LBP'yi hesapla
        lbp = local_binary_pattern(gray_img_uint8, n_points, radius, method='uniform')
        
        # LBP histogramını hesapla ve normalize et
        # Uniform LBP için bin sayısı n_points + 2 = 10'dur.
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        
        lbp_features = hist.tolist()

        # --- GLCM Öznitelikleri ---
        # GLCM için mesafeler ve açılar
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # GLCM hesapla (levels=256 varsayılır)
        # GLCM, uint8 tipinde bir girdi bekler.
        # Görüntüdeki gri seviyelerini 64'e düşürerek hesaplama maliyetini azaltıyoruz.
        glcm = graycomatrix(gray_img_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        # Kontrast, Enerji, Korelasyon, ve Homojenlik özelliklerini çıkar
        contrast = graycoprops(glcm, 'contrast').flatten().tolist()
        energy = graycoprops(glcm, 'energy').flatten().tolist()
        correlation = graycoprops(glcm, 'correlation').flatten().tolist()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten().tolist()
        
        glcm_features = contrast + energy + correlation + homogeneity
        
        # --- Kenar Tespiti (Canny) Öznitelikleri ---
        edges = canny(gray_img_uint8, sigma=1.0)
        edge_density = edges.sum() / edges.size  # Kenar yoğunluğu
        
        # --- Köşe Tespiti (Harris) Öznitelikleri ---
        corners = corner_harris(gray_img_uint8)
        corner_density = (corners > corners.mean()).sum() / corners.size  # Köşe yoğunluğu
        
        edge_corner_features = [edge_density, corner_density]
        
        # --- Histogram Equalization Öznitelikleri ---
        # Kontrast iyileştirmesi için
        equalized = exposure.equalize_hist(gray_img_uint8)
        eq_mean = equalized.mean()
        eq_std = equalized.std()
        
        histogram_features = [eq_mean, eq_std]
        
        # LBP, GLCM, kenar, köşe ve histogram özniteliklerini birleştir
        texture_features = lbp_features + glcm_features + edge_corner_features + histogram_features
        
        return texture_features
        
    except Exception as e:
        print(f"Doku öznitelik çıkarma hatası: {e}")
        return None

def extract_all_features(image_path):
    """
    Hem renk hem de doku özniteliklerini birleştirir.
    """
    color_feats = extract_color_features(image_path)
    texture_feats = extract_texture_features(image_path)
    
    if color_feats is None or texture_feats is None:
        return None
        
    return color_feats + texture_feats

def extract_tl_features(image_path, model=None):
    """
    Transfer Learning (MobileNetV2 bottleneck) öznitelikleri çıkarır.
    Model parametresi sağlanmazsa, global olarak yüklenen MobileNetV2'yi kullanır.
    """
    try:
        # TensorFlow import
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        except ImportError:
            return None
        
        # Dosya varlığını kontrol et
        if not os.path.isfile(image_path):
            return None
        
        # Model yükle (ilk kez çalıştırılırsa)
        if model is None:
            try:
                from tensorflow.keras.applications import MobileNetV2
                model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', 
                                   input_shape=(224, 224, 3))
            except Exception:
                return None
        
        # Görseli yükle ve preprocess et
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Eğer 4 kanallı (RGBA) ise 3 kanala indir
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Normalizasyon
        img_array = preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Bottleneck özellikleri çıkar
        features = model.predict(img_batch, verbose=0)
        
        return features.flatten().tolist()
    except Exception as e:
        return None

def main():
    print("--- Öznitelik Çıkarma Başlatılıyor ---")
    
    all_features = []
    all_labels = []
    
    # Veri setinin ana dizini
    base_dir = os.path.join(os.getcwd(), DATASET_PATH)
    
    # Sınıflar arasında döngü
    for label_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(base_dir, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Uyarı: Sınıf dizini bulunamadı: {class_dir}")
            continue
            
        print(f"Sınıf: {class_name} ({label_index}) işleniyor...")
        
        # Sınıf dizinindeki tüm görselleri al
        for filename in os.listdir(class_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, filename)
                
                # Öznitelikleri çıkar
                features = extract_all_features(image_path)
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label_index)
                else:
                    print(f"Görsel atlandı: {image_path}")
    
    # NumPy dizilerine dönüştür
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Öznitelik isimlerini tanımla (Renk + Doku)
    # RGB istatistikleri - 12 öznitelik
    color_names = ['R_Mean', 'R_Std', 'R_Min', 'R_Max',
                   'G_Mean', 'G_Std', 'G_Min', 'G_Max',
                   'B_Mean', 'B_Std', 'B_Min', 'B_Max']
    
    # HSV istatistikleri - 6 öznitelik
    hsv_names = ['H_Mean', 'H_Std', 'S_Mean', 'S_Std', 'V_Mean', 'V_Std']
    
    # LBP öznitelik isimleri (Uniform LBP için 10 bin)
    lbp_names = [f'LBP_Bin_{i}' for i in range(10)]
    
    # GLCM öznitelik isimleri (4 açı için Kontrast, Enerji, Korelasyon, Homojenlik)
    angles = ['0', '45', '90', '135']
    glcm_contrast_names = [f'GLCM_Contrast_{a}' for a in angles]
    glcm_energy_names = [f'GLCM_Energy_{a}' for a in angles]
    glcm_correlation_names = [f'GLCM_Correlation_{a}' for a in angles]
    glcm_homogeneity_names = [f'GLCM_Homogeneity_{a}' for a in angles]
    
    # Kenar ve köşe özniteikleri
    edge_corner_names = ['Edge_Density', 'Corner_Density']
    
    # Histogram equalization öznitelikleri
    histogram_names = ['Equalized_Mean', 'Equalized_Std']
    
    feature_names = (color_names + hsv_names + lbp_names + glcm_contrast_names + 
                     glcm_energy_names + glcm_correlation_names + glcm_homogeneity_names + 
                     edge_corner_names + histogram_names)
    
    # Veri setini kaydet
    data = {'X': X, 'y': y, 'feature_names': feature_names, 'classes': CLASSES}
    with open('features_combined.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\nToplam {len(X)} görselden renk ve doku öznitelikleri çıkarıldı.")
    print(f"Öznitelik boyutu: {X.shape[1]}")
    print("Veri 'features_combined.pkl' dosyasına kaydedildi.")

if __name__ == '__main__':
    # Sanal ortamı etkinleştir
    # Bu betik, sanal ortamın içinden çalıştırılacağı varsayımıyla yazılmıştır.
    # Terminalde çalıştırma komutu: ./venv/bin/python flower_classification_project/feature_extraction.py
    
    # Ana fonksiyonu çağır
    main()
    
    # Çalışma dizinini geri değiştir
    os.chdir('..')

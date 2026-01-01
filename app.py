import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
# feature_extraction.py dosyasından gerekli fonksiyonları içe aktarır
from feature_extraction import extract_color_features, extract_texture_features, extract_tl_features, CLASSES, IMAGE_SIZE
import json

# --- Flask Uygulama Ayarları ---
app = Flask(__name__)
# Yüklenen görselleri statik olarak erişilebilir kılmak için
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Session için GEREKLİ. Lütfen bu anahtarı kimseyle paylaşmayın!
app.config['SECRET_KEY'] = 'sınıflandırma_arayüzü_gizli_anahtarı_cok_guclu' 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FLOWER_CLASSES = CLASSES 

# İngilizce sınıf adlarını Türkçe karşılıklarıyla eşleyen sözlük
TURKISH_NAMES = {
    'daisy': 'Papatya',
    'dandelion': 'Karahindiba',
    'roses': 'Gül',
    'sunflowers': 'Ayçiçeği',
    'tulips': 'Lale'
}

# Doğruluk oranları (Grafik için kullanılır)
# TL (Transfer Learning) model using MobileNetV2 bottleneck features
MODEL_ACCURACIES = {
    "İyileştirilmiş SVM (MobileNetV2)": 0.9428,
    "Temel SVM (RBF Kernel)": 0.6965,
    "KNN (K=5)": 0.6040,
    "Naive Bayes": 0.5286
}

# Ana modelin `MODELS` içinde kullanılan anahtar adı (tek bir yerden yönetmek için)
MAIN_MODEL_KEY = 'İyileştirilmiş SVM (MobileNetV2)'
# --- Global Model ve Ölçekleyici Yükleme ---
MODELS = {}
TL_MODEL = None
TL_MODEL_ACTIVE = False
try:
    # Eğer İyileştirilmiş Transfer Learning (TL) model mevcutsa onu tercih et; yoksa fallback yapılır
    if os.path.exists('tl_improved_svm_model.pkl') and os.path.exists('tl_improved_scaler.pkl'):
        try:
            TL_MODEL = pickle.load(open('tl_improved_svm_model.pkl', 'rb'))
            MODELS['İyileştirilmiş SVM (MobileNetV2)'] = TL_MODEL
            SCALER = pickle.load(open('tl_improved_scaler.pkl', 'rb'))
            TL_MODEL_ACTIVE = True
            print("İyileştirilmiş Transfer Learning (MobileNetV2 + Negatif Örnekler) modeli yüklendi.")
        except Exception as e:
            print(f"TL model yükleme hatası: {e}. Fallback yapılıyor...")
            TL_MODEL_ACTIVE = False
    elif os.path.exists('tl_svm_model.pkl') and os.path.exists('tl_scaler.pkl'):
        try:
            TL_MODEL = pickle.load(open('tl_svm_model.pkl', 'rb'))
            MODELS['İyileştirilmiş SVM (MobileNetV2)'] = TL_MODEL
            SCALER = pickle.load(open('tl_scaler.pkl', 'rb'))
            TL_MODEL_ACTIVE = True
            print("Transfer Learning (MobileNetV2) modeli yüklendi.")
        except Exception as e:
            print(f"TL model yükleme hatası: {e}. Fallback yapılıyor...")
            TL_MODEL_ACTIVE = False
    # Fallback: TL model yoksa veya hata aldıysa, eski modeli kullan
    if not TL_MODEL_ACTIVE:
        if os.path.exists('best_svm_random_tuned.pkl'):
            MODELS['İyileştirilmiş SVM (MobileNetV2)'] = pickle.load(open('best_svm_random_tuned.pkl', 'rb'))
        else:
            MODELS['İyileştirilmiş SVM (MobileNetV2)'] = pickle.load(open('best_svm_model.pkl', 'rb'))
        SCALER = pickle.load(open('scaler.pkl', 'rb'))
        print("Eski SVM modeli ve scaler yüklendi (TL model bulunamadı).")

    # Diğer modelleri yükle (uyum olursa)
    MODELS['Temel SVM (RBF Kernel)'] = pickle.load(open('svm_(rbf_kernel)_model.pkl', 'rb'))
    MODELS['KNN (K=5)'] = pickle.load(open('knn_(k=5)_model.pkl', 'rb'))
    MODELS['Naive Bayes'] = pickle.load(open('naive_bayes_model.pkl', 'rb'))
    
    print("Tüm Modeller ve Ölçekleyici başarıyla yüklendi.")
except FileNotFoundError as e:
    SCALER = None
    print(f"HATA: Model veya scaler yüklenemedi: {e}.")
except Exception as e:
    SCALER = None
    print(f"Beklenmeyen hata ile model yüklenemedi: {e}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_flower(image_path):
    """Görselden öznitelikleri çıkarır ve tüm modellerle tahmin yapar."""
    
    if not SCALER or not MODELS:
        return {"error": "Model yüklenemedi. Tahmin yapılamıyor."}

    # 1. Eğer TL model aktif ise MobileNetV2 öznitelikleri çıkar, değilse eski öznitelikleri kullan
    if TL_MODEL_ACTIVE:
        features = extract_tl_features(image_path)
        if features is None:
            # TL model başarısız olursa, tüm modellere "Tahmin edilemedi" dön
            # (yani tüm modellerin tahminini skip et)
            return {
                "main_prediction": "Tahmin edilemedi",
                "all_predictions": {name: "Tahmin edilemedi" for name in MODELS.keys()},
                "skip_other_models": True
            }
        combined_features = np.array(features).reshape(1, -1)
    else:
        # Öznitelik Çıkarma (eski yol)
        color_feats = extract_color_features(image_path, target_size=IMAGE_SIZE)
        texture_feats = extract_texture_features(image_path, target_size=IMAGE_SIZE)

        if color_feats is None or texture_feats is None:
            return {"error": "Model bu görselin bir çiçek resmi olmadığını tespit etti. Bulunamadı."}
        
        combined_features = np.array(color_feats + texture_feats).reshape(1, -1)
    
    # 2. Ölçekleme
    X_scaled = SCALER.transform(combined_features)
    
    # 2.5. TL modeli için non-flower check (çiçek olmayan resimleri filtrele)
    if TL_MODEL_ACTIVE:
        try:
            tl_predictions = TL_MODEL.predict_proba(X_scaled)[0]
            # Debug: print top probabilities for inspection
            try:
                print(f"TL predict_proba (top): {np.round(tl_predictions,3)}")
            except Exception:
                pass
            predicted_class = np.argmax(tl_predictions)
            max_confidence = tl_predictions[predicted_class]
            
            # Eğer model %90+ emin olarak "çiçek değil" (Non-Flower = sınıf 5) tahmin ediyorsa, reddet
            # VEYA çiçek sınıflarından (0-4) birine düşük güvenle tahmin ediyorsa (daha sıkı eşik)
            is_non_flower_confident = (predicted_class == 5 and max_confidence >= 0.90)
            is_flower_uncertain = (predicted_class < 5 and max_confidence < 0.65)
            
            if is_non_flower_confident or is_flower_uncertain:
                return {
                    "main_prediction": "Tahmin edilemedi",
                    "all_predictions": {name: "Tahmin edilemedi" for name in MODELS.keys()},
                    "skip_other_models": True
                }
        except Exception:
            pass  # Confidence kontrolü yapılamadıysa devam et
    
    # 3. Tüm Modellerle Tahmin
    prediction_results = {}
    incompatible_models = []
    model_errors = {}

    for name, model in MODELS.items():
        expected = getattr(model, 'n_features_in_', None)
        # Eğer model beklenen öznitelik sayısı ile uyumluysa direkt kullan
        if expected is None or expected == X_scaled.shape[1]:
            try:
                prediction_index = model.predict(X_scaled)[0]
                try:
                    prediction_label = FLOWER_CLASSES[prediction_index]
                except Exception:
                    prediction_label = str(prediction_index)
                prediction_results[name] = prediction_label
            except Exception as e:
                model_errors[name] = str(e)
            continue

        # Eğer uyumsuzsa, deneme amaçlı combined_features'in ilk `expected` özniteliğini kullan
        try:
            # combined_features before scaling
            reduced = combined_features.flatten()[:expected]
            # Eğer scaler mean_ ve scale_ varsa, ölçeklemede aynı scaler'ın ilgili dilimini kullan
            if hasattr(SCALER, 'mean_') and hasattr(SCALER, 'scale_'):
                mean_slice = SCALER.mean_[:expected]
                scale_slice = SCALER.scale_[:expected]
                X_reduced_scaled = ((reduced - mean_slice) / scale_slice).reshape(1, -1)
            else:
                # Fallback: min-max normalize yerine ham veriyi kullan
                X_reduced_scaled = reduced.reshape(1, -1)

            prediction_index = model.predict(X_reduced_scaled)[0]
            try:
                prediction_label = FLOWER_CLASSES[prediction_index]
            except Exception:
                prediction_label = str(prediction_index)
            prediction_results[name] = prediction_label
        except Exception as e:
            # Eğer yine hata alırsak, kaydet ve atla
            model_errors[name] = str(e)

    # Eğer ana model uyumsuz veya hata veriyorsa, kullanıcıya geri bildir
    if MAIN_MODEL_KEY not in prediction_results:
        # Öncelikli hata mesajlarından birini oluştur
        if any(m.get('model') == MAIN_MODEL_KEY for m in incompatible_models):
            return {"error": f"Ana model özellik uyumsuzluğu: {MAIN_MODEL_KEY} beklenen özellik sayısı farklı."}
        elif MAIN_MODEL_KEY in model_errors:
            return {"error": f"Ana model tahmin hatası: {model_errors[MAIN_MODEL_KEY]}"}
        else:
            return {"error": "Ana modelden tahmin alınamadı (uyumsuzluk veya hata)."}

    main_prediction = prediction_results[MAIN_MODEL_KEY]

    result = {
        "main_prediction": main_prediction,
        "all_predictions": prediction_results
    }

    if incompatible_models:
        result['incompatible_models'] = incompatible_models
    if model_errors:
        result['model_errors'] = model_errors

    return result


# --- WEB YOLLARI ---

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        # ... Dosya yükleme kontrolü ...
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename): return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            file.save(filepath)
            
            # 1. Önceki tahmini (mevcut olanı) oturumda 'previous' olarak kaydet
            if 'current_prediction' in session:
                session['previous_prediction'] = session['current_prediction']
            else:
                session['previous_prediction'] = None

            # 2. Yeni tahmin yap
            result = predict_flower(filepath)

            # Eğer tahmin sırasında hata döndüyse, kullanıcıya anlamlı mesaj göster
            if isinstance(result, dict) and result.get('error'):
                # Kayıtlı görselin önizlemesini göstermek için current_prediction içine image_url kaydet
                session['current_prediction'] = {
                    'image_url': url_for('static', filename=f'uploads/{filename}'),
                    'invalid_image': True
                }

                # Daha kullanıcı-dostu bir hata mesajı üret
                err = result.get('error', '')
                if 'Resim RGB format' in err or 'Görsel dosyası bulunamad' in err or 'Öznitelik çıkarma başarısız' in err:
                    session['error_message'] = 'Bu görsel bir çiçek resmi gibi görünmüyor veya geçersiz. Lütfen farklı bir görsel deneyin.'
                else:
                    session['error_message'] = result.get('error')

                return redirect(url_for('index'))

            # 3. Tahmin sonuçlarını Türkçe'ye çevir
            # Eğer skip_other_models flag'i varsa, tüm modelleri skip et
            if result.get('skip_other_models'):
                tr_all_preds = {name: "Tahmin edilemedi" for name in MODELS.keys()}
            else:
                tr_all_preds = {}
                all_preds = result.get('all_predictions', {})
                # Tüm modeller için bir giriş oluştur; uyumsuz/atlanan modeller için yer tutucu göster
                for model_name in MODELS.keys():
                    if model_name in all_preds:
                        en_pred = all_preds[model_name]
                        # Eğer "Tahmin edilemedi" ise olduğu gibi döndür
                        if en_pred == "Tahmin edilemedi":
                            tr_all_preds[model_name] = "Tahmin edilemedi"
                        else:
                            tr_all_preds[model_name] = TURKISH_NAMES.get(en_pred.lower(), en_pred)
                    else:
                        tr_all_preds[model_name] = 'Tahmin edilemedi'

            tr_main_pred = TURKISH_NAMES.get(result['main_prediction'].lower(), result['main_prediction'])

            # Not: Uyumsuz modeller veya model hataları artık kullanıcıya uyarı olarak gösterilmeyecek.
            
            # 4. Şu anki veriyi Türkçe olarak oturuma kaydet ('current' olarak)
            current_prediction = {
                'image_url': url_for('static', filename=f'uploads/{filename}'), 
                'main_prediction': tr_main_pred, # Türkçe ana tahmin
                'all_predictions': tr_all_preds # Türkçe tüm tahminler
            }
            session['current_prediction'] = current_prediction
            
            # Flask'in cache sorunlarını aşmak için redirect kullan
            return redirect(url_for('index'))
    
    # GET isteği veya redirect sonrası (veriyi session'dan alır)
    prediction_data = session.get('current_prediction')
    previous_prediction = session.get('previous_prediction')

    image_url = prediction_data['image_url'] if prediction_data else None
    
    # Doğruluk oranlarını JSON'a çevir (grafik için)
    accuracy_json = json.dumps(MODEL_ACCURACIES)

    # Hata mesajını oturumdan al ve şablona aktar
    error_message = session.pop('error_message', None)
    
    return render_template('index.html', 
                           prediction_data=prediction_data, 
                           image_url=image_url, 
                           accuracy_json=accuracy_json,
                           previous_prediction=previous_prediction,
                           error_message=error_message)


if __name__ == '__main__':
    # Gerekli klasörleri oluştur
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    # Run without the automatic reloader/debugger to avoid TensorFlow-triggered restarts
    app.run(debug=False, use_reloader=False)
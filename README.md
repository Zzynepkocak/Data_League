# 🏥 Randevuya Gelmeme (No-Show) Tahmini

Bu not defteri, bir hastanın randevusuna **gelmeyip gelmeyeceğini** tahmin eden bir makine öğrenmesi hattı (pipeline) içerir. Randevu, hasta ve klinik tabloları birleştirilir, özellik mühendisliği yapılır ve **LightGBM + XGBoost** (CatBoost isteğe bağlı) modelleri bir ensemble ile birleştirilir.

**Dosya:** `zzynep-2.ipynb` (orijinal, düzeltilmemiş sürüm)

---

## 🎯 Amaç

| | |
|---|---|
| **Hedef değişken** | `label_noshow` (1 = gelmedi, 0 = geldi) |
| **Problem türü** | İkili sınıflandırma (binary classification) |
| **Değerlendirme metriği** | PR-AUC (`average_precision_score`) |
| **Çıktı** | Test randevuları için gelmeme olasılığı (`submission_*.csv`) |

---

## 📦 Veri Dosyaları

Not defteri aşağıdaki CSV dosyalarının **not defteriyle aynı klasörde** olmasını bekler:

| Dosya | İçerik |
|---|---|
| `appointments_train.csv` | Eğitim randevuları (hedef sütunu `label_noshow` dahil) |
| `appointments_test.csv` | Tahmin yapılacak randevular |
| `patients.csv` | Hasta bilgileri (`patient_id` ile birleşir) |
| `clinics.csv` | Klinik bilgileri (`clinic_id` ile birleşir) |
| `sample_submission.csv` | Teslim dosyası şablonu |

Kodun kullandığı başlıca sütunlar: `appointment_datetime`, `booking_datetime`, `lead_time_hours`, `sms_lead_hours`, `wait_mins_est`, `booking_channel`, `prior_noshow_rate`, `ses_score`, `distance_km`, `residence_lat/lon`, `clinic_lat/lon`, `specialty`.

---

## 🛠️ Gereksinimler

```bash
pip install numpy pandas scikit-learn lightgbm xgboost
pip install catboost   # isteğe bağlı (kodda şu an kapalı)
```

`StratifiedGroupKFold` için scikit-learn ≥ 1.0 önerilir; yoksa kod `StratifiedKFold`'a geçer.

---

## ▶️ Çalıştırma

1. CSV dosyalarını not defteriyle aynı klasöre koyun.
2. `zzynep-2.ipynb` dosyasını Jupyter, Colab veya Kaggle'da açın.
3. Hücreleri sırayla çalıştırın.

> ⚠️ Kaggle'da veriler `/kaggle/input/<yarışma-adı>/` altında durduğu için, dosya yolları değiştirilmeden çalıştırıldığında ilk hücre `FileNotFoundError` verir (kaydedilmiş çıktıda da bu hata görünüyor).

---

## 🔍 Not Defterinin Akışı

### 1. Kurulum
Kütüphaneler içe aktarılır, `RANDOM_STATE = 42` ve `TARGET = "label_noshow"` tanımlanır.

### 2. Veri Yükleme ve Birleştirme
Randevu tabloları `patients` (`patient_id`) ve `clinics` (`clinic_id`) ile **left join** yapılır. Klinikteki `specialty` sütunu `clinic_specialty` olarak yeniden adlandırılır.

### 3. Özellik Mühendisliği (`add_all_features`)
| Fonksiyon | Üretilen özellikler |
|---|---|
| `add_time_features` | Randevu/rezervasyon yıl, ay, gün, saat, haftanın günü; hafta sonu, pazartesi, cuma, sabah, öğle, akşam bayrakları; `hour_group`; iki tarih arasındaki gün/saat/dakika farkı; `lead_time` türevleri (gün, hafta, kısa/çok kısa/uzun bekleme) |
| `add_sms_features` | SMS gönderilip gönderilmediği, SMS'in gün cinsinden zamanı, geç/erken SMS bayrakları |
| `add_geographical_features` | Hasta ile klinik arasındaki enlem/boylam farkı ve Manhattan mesafesi |
| `add_ratio_and_interaction_features` | `time_pressure`, `behavior_score`, mesafe × bekleme süresi, sosyoekonomik skor × geçmiş gelmeme oranı vb. etkileşimler |
| `add_combination_categoricals` | Klinik×gün, kanal×saat, uzmanlık×gün, uzmanlık×saat grubu birleşik kategorileri |

### 4. Kodlama (Encoding)
- **Frekans kodlama:** Kimlik ve kategorik sütunların eğitim setindeki görülme sayısı (`*_freq`).
- **Target encoding:** Aynı sütunlar için 5 katlı OOF, smoothing'li (`alpha = 20`) ortalama hedef değeri (`*_te`).

### 5. Eksik Değerler ve Kategoriler
Sayısal sütunlar eğitim medyanıyla, kategorik sütunlar `"missing"` ile doldurulur. Train ve test kategorileri ortak bir kümeye hizalanıp `pd.Categorical` yapılır.

### 6. Çapraz Doğrulama
`patient_id` gruplu **StratifiedGroupKFold** (5 kat) kullanılır; böylece aynı hasta hem eğitimde hem doğrulamada yer almaz.

### 7. Modelleme
| Model | Önemli ayarlar |
|---|---|
| LightGBM | 1000 ağaç, `learning_rate=0.02`, `num_leaves=63`, 150 turluk early stopping |
| XGBoost | 1000 ağaç, `learning_rate=0.025`, `max_depth=6`, `eval_metric="aucpr"` |
| CatBoost | Parametreleri tanımlı, ancak eğitim kodu yorum satırına alınmış |

Her katta modellerin PR-AUC skorları ve LightGBM özellik önemleri kaydedilir.

### 8. Sonuçlar ve Teslim Dosyaları
Model bazında katlara göre ve OOF PR-AUC yazdırılır, en önemli 40 özellik listelenir. Şu dosyalar oluşturulur:
- `submission_lgbm_enhanced.csv`
- `submission_xgb_enhanced.csv`
- `submission_catboost_enhanced.csv`
- `submission_ensemble_enhanced.csv` (`0.40·LGBM + 0.30·XGB + 0.30·CatBoost`)

---

## 🐞 Bilinen Sorunlar

- **Dosya yolu:** CSV'ler göreli yolla okunduğu için Kaggle'da `FileNotFoundError` alınıyor.
- **CatBoost devre dışı ama hesaba katılıyor:** CatBoost eğitimi kapalı olduğu için `cat_oof` ve `cat_test_preds` sıfır kalıyor. Fold döngüsündeki ensemble bunu hesaba katıyor, fakat **sonuç ve teslim hücreleri hâlâ `0.30 * cat` kullanıyor**. Sonuç olarak:
  - genel ensemble skoru ve `submission_ensemble_enhanced.csv` tahminleri bozuluyor (olasılıklar %30 aşağı çekiliyor);
  - `submission_catboost_enhanced.csv` tamamen sıfırlardan oluşuyor;
  - `np.mean(cat_scores)` boş liste yüzünden `NaN` yazdırıyor.
- **Target encoding sızıntısı:** Target encoding kendi `StratifiedKFold` bölünmesini kullanıyor, model ise `StratifiedGroupKFold` kullanıyor. Fold'lar uyuşmadığı için doğrulama satırlarının hedef bilgisi eğitim özelliklerine sızıyor ve **CV skoru olduğundan iyi görünebiliyor**.
- **`patient_id` özellik olarak kullanılıyor:** Grup CV'de doğrulama hastaları eğitimde hiç bulunmadığı için ham `patient_id` ve onun target encoding'i genelleme yerine ezberlemeye yol açıyor.
- **XGBoost'ta early stopping yok:** Her katta 1000 ağacın tamamı eğitiliyor.
- **Teslim dosyası sıraya güveniyor:** Tahminler `sample_submission` sırasıyla değil, test satır sırasıyla yazılıyor; `appointment_id` üzerinden eşleştirme yapılmıyor.
- **Küçük sorunlar:** Çıplak `except:` kullanımı; koşulsuz `!pip install catboost` hücresi.

---

## 🚀 Geliştirme Önerileri

- Veri klasörünü otomatik bulmak (`/kaggle/input/**` veya yerel klasör).
- CatBoost'u isteğe bağlı yapmak ve ensemble ağırlıklarını yalnızca eğitilen modellere göre normalize etmek.
- Target encoding'de modelle aynı fold'ları kullanmak; `patient_id`'yi özelliklerden çıkarmak.
- XGBoost'a `early_stopping_rounds` eklemek.
- Teslim dosyasını `appointment_id` üzerinden birleştirmek.
- Ensemble ağırlıklarını OOF tahminleri üzerinde optimize etmek; eşik ve kalibrasyon analizi yapmak.

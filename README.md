# Submission 1: Churn Prediction Pipeline with TFX

**Nama**: Dzikri Maulana  
**Username dicoding**: dzikrimaulana87

|                        | Deskripsi                                                                                                                                                                                                 |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Dataset**            | [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)                                                                                                          |
| **Masalah**            | Memprediksi apakah seorang pelanggan akan berhenti (churn) berdasarkan informasi demografi dan layanan yang digunakan. Ini penting untuk membantu perusahaan mempertahankan pelanggan.                |
| **Solusi machine learning** | Membangun pipeline machine learning terotomatisasi menggunakan TFX (TensorFlow Extended) yang memuat, memproses, dan melatih model klasifikasi churn secara end-to-end, kemudian mendeloynya sebagai REST API. |
| **Metode pengolahan**  | Dataset diproses menggunakan komponen TFX seperti: <br>- `CsvExampleGen` untuk ingesting data<br>- `StatisticsGen` dan `SchemaGen` untuk inferensi awal<br>- `Transform` untuk normalisasi numerik (`tft.scale_to_0_1`) dan one-hot encoding untuk fitur kategorikal (`tft.compute_and_apply_vocabulary`) |
| **Arsitektur model**   | Model Keras Sequential: terdiri dari beberapa Dense layer dengan ReLU, dan satu output Sigmoid. Input model berupa dictionary dari 18 fitur transformasi berakhiran `_xf`. Pipeline dijalankan menggunakan Apache Beam (DirectRunner). |
| **Metrik evaluasi**    | `Accuracy`digunakan sebagai metrik utama selama pelatihan dan evaluasi menggunakan TFX `Evaluator`.                                                                                          |
| **Performa model**     | Model menunjukkan akurasi +- **80%**, hasil ini dinilai memadai untuk keperluan prediksi churn pelanggan.                                                        |
| **Opsi deployment**    | Model disajikan dalam bentuk REST API menggunakan Flask yang dikemas dalam Docker container dan dideploy ke Google Cloud Platform (GCP) menggunakan Cloud Run. API melayani endpoint `/predict` dan `/metrics`. |
| **Web app**            | https://churn-detection-124339833986.asia-southeast1.run.app/                                                                                              |
| **Monitoring**         | Endpoint `/metrics` diekspos untuk Prometheus. Monitoring dilakukan dengan Prometheus dalam container terpisah yang diset untuk scraping API setiap 15 detik. Visualisasi dan metrik dapat diakses dari UI Prometheus lokal (`http://localhost:9090`). |

> catatan: berikut contoh input untuk endpoint: https://churn-detection-124339833986.asia-southeast1.run.app/predict
> 
> {
  "tenure_xf": [[0.4]],
  "MonthlyCharges_xf": [[0.3]],
  "TotalCharges_xf": [[0.5]],
  "gender_xf": [[0.0, 1.0]],
  "Partner_xf": [[1.0, 0.0]],
  "Dependents_xf": [[1.0, 0.0]],
  "PhoneService_xf": [[1.0, 0.0]],
  "MultipleLines_xf": [[0.0, 1.0, 0.0]],
  "InternetService_xf": [[0.0, 1.0, 0.0]],
  "OnlineSecurity_xf": [[1.0, 0.0, 0.0]],
  "OnlineBackup_xf": [[0.0, 0.0, 1.0]],
  "DeviceProtection_xf": [[0.0, 0.0, 1.0]],
  "TechSupport_xf": [[1.0, 0.0, 0.0]],
  "StreamingTV_xf": [[0.0, 1.0, 0.0]],
  "StreamingMovies_xf": [[0.0, 1.0, 0.0]],
  "Contract_xf": [[1.0, 0.0, 0.0]],
  "PaperlessBilling_xf": [[1.0, 0.0]],
  "PaymentMethod_xf": [[0.0, 0.0, 1.0, 0.0]]
}
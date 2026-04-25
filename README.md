## Prediksi Harga Rumah Jakarta (Solo Project)

Aplikasi **Streamlit** untuk memprediksi **harga rumah** (regresi) dan **kategori harga** (klasifikasi) berbasis model Machine Learning.

### Cara menjalankan (Windows / PowerShell)

Masuk ke folder project ini, lalu:

```powershell
# aktifkan virtualenv (kalau pakai venv yang ada di repo)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

### File penting

- `app.py`: aplikasi Streamlit
- `house_price_regression.pkl`: model prediksi harga (regresi)
- `house_price_classification.pkl`: model kategori harga (klasifikasi)
- `model_columns.pkl`: daftar fitur/kolom yang dipakai model

### Author

Husain Abdul Ghani

### Catatan versi model (penting)

Model `.pkl` di project ini dibuat dengan **scikit-learn 1.6.1**. Karena itu `requirements.txt` sudah dipin ke versi tersebut supaya tidak muncul peringatan mismatch saat run.

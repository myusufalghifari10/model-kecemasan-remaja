# Model Risiko Kecemasan Remaja

Prediksi skor kecemasan siswa SMK dari 88 fitur kuesioner: 58 item mentah (18 item faktor
tekanan, 18 coping, 10 perilaku sehat, 12 dukungan lingkungan), 13 skor faktor, dan 4
variabel demografi. Target `y_kontinu` = rata-rata 20 item kecemasan (skala 1–4).

Data: 306 siswa, survei cross-sectional. Lihat `docs/codebook.md`.

## Model

Dua tahap, keduanya dilatih **di dalam lipatan** pada data latih lipatan saja:

1. **Tulang punggung linier** — `StandardScaler` + `PLSRegression(n_components=1)`.
   Satu komponen PLS adalah arah yang memaksimalkan kovarians dengan y: poros "distres
   umum" yang stabil pada 88 fitur Likert yang saling berkorelasi. Dengan begitu pohon
   hanya perlu mempelajari simpangannya, bukan memprediksi y dari nol.
2. **XGBoost** (`objective = reg:absoluteerror`) pada residual `y − base`, dengan bobot per
   bin skor kecemasan (bobot populasi, cap 3,0).

Prediksi akhir = `base + XGB`.

## Hasil

| metrik | CV 5×6 (306 siswa) | Latih 238 → uji 68 |
|---|---|---|
| **MAE** | **0,2165** (CI95 0,196–0,237) | 0,2407 mentah / **0,2230** berbobot populasi |
| **R²** | **0,6305** | 0,6884 |

Sebagai pembanding, analisis regresi berganda sebelumnya (9 faktor) mencapai R² 0,398 secara
in-sample. Model ini mencapai R² 0,6305 secara **out-of-fold**, tetapi dengan 88 prediktor
— bukan perbandingan setara, hanya penanda besaran.

Diagnosis per bin (CV): galat terkonsentrasi di ekor. Bin 9 (n=6) MAE 0,415 dengan bias
+0,383; bin 8 (n=9) MAE 0,291 dengan bias +0,291. Model menaksir terlalu rendah pada siswa
berisiko tinggi.

## 30 model untuk mengukur, 6 model untuk produksi

- **Pengukuran** — CV 5 lipat × 6 ulangan = 30 model. Setiap siswa diprediksi oleh 6 model
  yang tidak pernah melihatnya; 1.836 prediksi dirata-ratakan menjadi 306 prediksi. Dari
  sinilah angka 0,2165 berasal.
- **Produksi** — 6 model (satu per ulangan), jawaban dirata-ratakan. Konfigurasi ini persis
  yang diukur, jadi 0,2165 berlaku untuk artefak yang dikirim.

## Label kemungkinan stress

`--mode predict` menerima data siswa baru dan mengeluarkan `skor` + `status`:
**Kemungkinan stress** bila skor > 2,5, selain itu **Tidak**.

Ambang 2,5 mengikuti skema kategori pada analisis sebelumnya, dan skema itu dapat
direproduksi persis dari data ini:

| kategori | batas | n |
|---|---|---|
| Rendah | ≤ 1,75 | 115 |
| Sedang | 1,75 – 2,50 | 153 |
| Tinggi | 2,50 – 3,25 | 35 |
| Sangat Tinggi | > 3,25 | 3 |

Karena nilai tepat 2,50 masih masuk Sedang, "Tinggi" mulai **di atas** 2,5 → 38 siswa
(12,4%). Batas ini dipakai apa adanya, jadi label diturunkan dari skor prediksi yang
cenderung menaksir rendah di ekor: pada CV hanya 22 dari 43 siswa berisiko tinggi yang
tertandai (sensitivitas 51%), dengan spesifisitas 94%. Label ini penanda awal, bukan
penentuan status.

## Penjelasan per siswa (SHAP)

`--mode explain` menjawab "kenapa siswa ini ditandai": prediksi dipecah menjadi kontribusi
tiap fitur. Karena model = `base + XGB` dan keduanya aditif:

- bagian **PLS** linier → kontribusi eksak `koefisien × nilai terstandardisasi`;
- bagian **XGBoost** pada residual → TreeSHAP bawaan XGBoost (`pred_contribs`), tanpa
  library tambahan.

Jumlah semua kontribusi + bias = skor prediksi persis (diperiksa setiap run, selisih
~2e-07), dan kontribusi dirata-ratakan atas 6 model bag.

```bash
python src/train.py --mode explain --input siswa_baru.csv          # 5 fitur teratas
python src/train.py --mode explain --input siswa_baru.csv --top 10
```

Keluaran `outputs/model/prediksi_penjelasan.csv`: `row_id, skor, status, fitur_1..n,
kontribusi_1..n`. Kontribusi positif menaikkan skor, negatif menurunkan. Ini menjelaskan
model, bukan sebab-akibat.

## Isi repo

```
src/train.py            model + protokol evaluasi + inferensi (7 mode)
src/config.py           nama kolom, path, seed
outputs/model/cv.json   hasil CV populasi 306 (MAE, R², per bin)
outputs/model/per_bin.csv
outputs/model/test_1d5f63ef.json   konfirmasi pada split 238→68
outputs/model/produksi.json        spesifikasi artefak produksi
data/split_v1.json      split terkunci 238/68 (sha 1d5f63efb7d61488)
docs/codebook.md        definisi variabel dan aturan pembersihan data
```

Tidak ikut repo (berisi data responden atau berkas besar): `data/raw/`,
`data/features.parquet`, `outputs/model/oof.csv`, `outputs/model/model_bag.joblib`.

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python src/train.py --mode check-split   # verifikasi split terkunci
python src/train.py --mode self-test     # uji fungsi murni
python src/train.py --mode cv            # CV 5x6 -> outputs/model/cv.json
python src/train.py --mode test          # 238 -> 68 (konfirmasi)
python src/train.py --mode final         # bag 6 model -> model_bag.joblib
python src/train.py --mode predict --input siswa_baru.csv   # skor + label
python src/train.py --mode explain --input siswa_baru.csv   # skor + label + fitur teratas
```

`--mode predict` membutuhkan CSV/parquet dengan 76 kolom input model (72 numerik + 4
kategorikal) memakai nama kolom yang sama seperti `data/features.parquet`; kolom `row_id`
opsional. Keluaran default `outputs/model/prediksi.csv`.

`--mode cv` dan `--mode final` membutuhkan `data/features.parquet` (306 × 82: 76 kolom
input → 88 setelah one-hot, plus target) yang tidak disertakan karena memuat jawaban per
siswa.

## Catatan pembacaan

- 0,2165 adalah estimasi **populasi**: himpunan evaluasi CV = seluruh 306 siswa, termasuk
  68 siswa uji, sehingga angka ini **bukan** angka uji.
- Angka 0,2407/0,2230 berasal dari split tetap 238→68 dan dilaporkan sekali sebagai
  konfirmasi, bukan untuk memilih model atau parameter.
- Bin 10 hanya 3 siswa (bin 9: 6, bin 8: 9). MAE per bin untuk bin dengan n ≤ 5 dilaporkan
  tetapi tidak boleh diranking.
- Label "Kemungkinan stress" adalah binerisasi skor prediksi pada ambang 2,5, bukan model
  klasifikasi tersendiri; performanya tidak dioptimalkan untuk keputusan itu.

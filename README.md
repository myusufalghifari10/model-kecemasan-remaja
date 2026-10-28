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
| **akurasi kategori** | **68,3%** (QWK 0,647) | 70,6% (QWK 0,730) |

Ketiga baris itu saja yang disimpan di `cv.json` dan `test_*.json`, ditambah tabel per bin
untuk menunjukkan kelemahan di ekor; metrik lain dihitung tidak dipersistenkan.

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

## Prediksi, kategori, dan alasan (satu keluaran)

`--mode predict` mengeluarkan tiga hal sekaligus, tanpa opsi tambahan:

1. **skor** — taksiran skor kecemasan (skala 1–4), rata-rata bag 6 model;
2. **kategori** — Rendah / Sedang / Tinggi / Sangat Tinggi;
3. **fitur_1..n + kontribusi_1..n** — fitur yang paling mendorong skor itu (default 5).

Skema kategori mengikuti analisis sebelumnya, dan dapat direproduksi persis dari data ini:

| kategori | batas | n |
|---|---|---|
| Rendah | ≤ 1,75 | 115 |
| Sedang | 1,75 – 2,50 | 153 |
| Tinggi | 2,50 – 3,25 | 35 |
| Sangat Tinggi | > 3,25 | 3 |

> Angka ambangnya **tidak tertulis** di dokumen analisis sebelumnya — di sana hanya ada
> jumlah per kategori. Batas 1,75 / 2,50 / 3,25 adalah satu-satunya pembagian berspasi
> sama pada skala 1–4 (0,75 per pita, setara skala jumlah 20–80 dengan interval 15) yang
> mereproduksi jumlah itu persis, jadi statusnya rekonstruksi, bukan kutipan.

Akurasi kategori pada CV (prediksi OOF tiap siswa dibanding kategorinya yang sebenarnya):
**68,3% tepat**, 31,7% meleset satu tingkat, **tidak ada yang meleset dua tingkat**,
QWK 0,647. Kategori tertinggi paling sulit ditebak (hanya 1 siswa diprediksi Sangat Tinggi
dari 3 yang sebenarnya), karena prediksi model sedikit terkompresi ke tengah.

### Cara kontribusinya dihitung

Model = `base + XGB`, dan keduanya aditif:

- bagian **PLS** linier → kontribusi eksak `koefisien × nilai terstandardisasi`;
- bagian **XGBoost** pada residual → TreeSHAP bawaan XGBoost (`pred_contribs`), tanpa
  library tambahan.

Jumlah semua kontribusi + bias = skor yang dilaporkan (diperiksa setiap run, selisih
~2e-07), dan kontribusinya dirata-ratakan atas 6 model bag. Kontribusi positif menaikkan
skor, negatif menurunkan. Ini menjelaskan model, bukan sebab-akibat; fitur yang saling
berkorelasi juga berbagi kredit secara arbitrer, jadi jangan menafsirkan selisih kecil
antar item sebagai fakta.

Keluaran `outputs/model/prediksi.csv`:
`row_id, skor, kategori, fitur_1, kontribusi_1, …, fitur_n, kontribusi_n`.

## Isi repo

```
src/train.py            model + protokol evaluasi + inferensi (6 mode)
src/config.py           nama kolom, path, seed
outputs/model/model_bag.joblib     artefak siap pakai: bag 6 model (6 MB)
outputs/model/cv.json   hasil CV populasi 306: MAE+CI95, R², akurasi kategori + QWK, per bin
outputs/model/per_bin.csv
outputs/model/test_1d5f63ef.json   konfirmasi pada split 238→68 (metrik yang sama)
outputs/model/produksi.json        spesifikasi artefak produksi + ambang kategori
data/split_v1.json      split terkunci 238/68 (sha 1d5f63efb7d61488)
docs/codebook.md        definisi variabel dan aturan pembersihan data
```

Tidak ikut repo — berisi jawaban per siswa atau data mentah: `data/raw/`,
`data/features.parquet`, `outputs/model/oof.csv`. Karena `data/features.parquet` tidak
ikut, `--mode cv` dan `--mode final` hanya bisa dijalankan oleh yang punya data mentah;
`--mode predict` tetap jalan dari hasil clone karena artefak model sudah ikut repo.

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python src/train.py --mode check-split   # verifikasi split terkunci
python src/train.py --mode self-test     # uji fungsi murni
python src/train.py --mode cv            # CV 5x6 -> outputs/model/cv.json
python src/train.py --mode test          # 238 -> 68 (konfirmasi)
python src/train.py --mode final         # bag 6 model -> model_bag.joblib
python src/train.py --mode predict --input siswa_baru.csv   # skor + label + alasan
```

`--mode predict` membutuhkan CSV/parquet dengan 76 kolom input model (72 numerik + 4
kategorikal) memakai nama kolom yang sama seperti `data/features.parquet`; kolom `row_id`
opsional. Keluaran default `outputs/model/prediksi.csv`.

Satu berkas bisa memuat berapa pun siswa: 5.000 siswa diproses dalam ~3,4 detik (bag 6
model + kontribusi fitur untuk semuanya). Rincian fitur di layar hanya dicetak untuk 3
siswa pertama supaya layar tidak penuh; seluruh siswa ada di berkas CSV keluaran.

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
- Kategori adalah pembagian skor prediksi pada ambang tetap, bukan model klasifikasi
  tersendiri; ambangnya tidak dioptimalkan untuk keputusan, jadi terbaca konservatif di
  ekor (Sangat Tinggi jarang diprediksi).

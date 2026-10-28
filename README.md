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

| metrik | CV 5×6 (306 siswa) |
|---|---|
| **MAE** | **0,2165** (CI95 0,196–0,237) |
| **R²** | **0,6305** |
| **akurasi kategori** | **69,3%** (QWK 0,734) |

Ketiga baris itu disimpan di `cv.json`, bersama tabel per bin (kelemahan
di ekor), tabel silang kategori, dan rincian per kategori (n, tepat, recall, MAE, bias
dengan konvensi `y − pred` seperti tabel per bin). Metrik lain tidak dipersistenkan.

Artefak yang dipakai untuk inference adalah bag 6 model di `model_bag.joblib`. Pada 306
siswa yang sudah dilihatnya, MAE-nya **0,1288** (R² 0,8502) — angka ini ikut dicatat di
`produksi.json` hanya sebagai penanda artefak, **bukan** performa. Harapan performa untuk
siswa baru tetap **0,2165**, karena pada CV setiap prediksi selalu berasal dari 6 model
yang belum pernah melihat siswa itu.

Sebagai pembanding, analisis regresi berganda sebelumnya (9 faktor) mencapai R² 0,398 secara
in-sample. Angka itu tidak bisa direproduksi dari data ini (16 kombinasi skor coping/dukungan
yang wajar memberi R² in-sample 0,473–0,566), jadi yang dihitung ulang adalah regresi 9 faktor
yang sama dengan protokol yang sama: MAE CV **0,2469**, R² CV **0,534** — lihat
`src/bandingkan_regresi.py` dan `outputs/model/perbandingan_regresi.json`. Varian terbaik itu
dipilih dari 16 kandidat pada lipatan yang sama, jadi keuntungannya ada di pihak regresi.

Diagnosis per bin (CV): galat terkonsentrasi di ekor. Bin 9 (n=6) MAE 0,415 dengan bias
+0,383; bin 8 (n=9) MAE 0,291 dengan bias +0,291. Model menaksir terlalu rendah pada siswa
berisiko tinggi.

## 30 model untuk mengukur, 6 model untuk produksi

- **Pengukuran** — CV 5 lipat × 6 ulangan = 30 model. Setiap siswa diprediksi oleh 6 model
  yang tidak pernah melihatnya; 1.836 prediksi dirata-ratakan menjadi 306 prediksi. Dari
  sinilah angka 0,2165 berasal.
- **Produksi** — 6 model: **#8, #11, #17, #20, #24, #26**. Dipilih dengan kriteria
  out-of-fold: tiap model dinilai hanya pada 61 siswa yang tidak dilihatnya, lalu MAE itu
  dibagi rata-rata MAE ulangannya untuk meredam tingkat kesulitan lipatan. Kriteria ini
  tidak memakai skor pada 306 siswa yang menjadi data latih model, jadi bukan pemilihan
  berdasarkan hafalan. Keuntungan pemilihan semacam ini terukur **±0,003 MAE** pada uji
  bersarang (memilih di separuh siswa buta, dinilai di separuh lain: 0,2454 vs 0,2485 untuk
  bag kanonik, menang 383/500) — kecil, dan estimasi performa untuk siswa baru tetap 0,2165.
  Dua catatan jujur: (a) komposisinya bukan "satu model per ulangan" — model #8/#11/#17/
  #20/#24/#26 berasal dari ulangan 1,2,3,3,4,5, jadi dua model dari ulangan 3 dan tidak ada
  dari ulangan 0; masing-masing tetap dilatih pada 244 dari 306 siswa (4/5). (b) Kriterianya
  memakai prediksi out-of-fold di seluruh 306 siswa, yaitu protokol yang sama dengan CV.

## Prediksi, kategori, dan alasan (satu keluaran)

`--mode predict` mengeluarkan tiga hal sekaligus, tanpa opsi tambahan:

1. **skor** — taksiran skor kecemasan (skala 1–4), rata-rata bag 6 model;
2. **kategori** — Rendah / Sedang / Tinggi;
3. **fitur_1..n + kontribusi_1..n** — fitur yang paling mendorong skor itu (default 5).

Skema kategori: tiga tingkat, dua ambang.

| kategori | batas | n |
|---|---|---|
| Rendah | ≤ 1,70 | 103 |
| Sedang | 1,70 – 2,20 | 115 |
| Tinggi | > 2,20 | 88 |

> Ambang 1,70 dan 2,20 dipilih dari grid dua parameter pada prediksi out-of-fold 306 siswa
> dengan kriteria **QWK, bukan akurasi**: memilih demi akurasi menghasilkan skema yang
> menumpuk 238 siswa di satu kelas (akurasi 0,83 tetapi QWK 0,56). Keduanya kira-kira
> rata-rata sampel (1,967) ∓ 0,5 simpangan baku (0,467). Uji bersarang, yaitu ambang dipilih
> di lipatan latih lalu dinilai di lipatan uji selama 30 lipatan, memberi **QWK 0,689 dan
> akurasi 66,0%**, jadi keunggulannya tidak hanya muncul di data yang dipakai memilih.
> Skema empat tingkat 115/153/35/3 dari dokumen analisis sebelumnya tetap dicatat sebagai
> pembanding, tetapi tidak dipakai: kelas teratasnya hanya 3 siswa dan tidak dapat
> dipelajari (recall 1 dari 3).

Akurasi kategori pada CV (prediksi OOF tiap siswa dibanding kategorinya yang sebenarnya):
**69,3% tepat**, 30,7% meleset satu tingkat, 1 siswa meleset dua tingkat, **QWK 0,734**.
Recall tiap tingkat 68% / 67% / 74%, jauh lebih seimbang daripada skema empat tingkat
(70% / 73% / 46% / 33%).

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
src/build_features.py   data mentah .sav -> matriks fitur (tahap 1, aturan cleaning)
src/bandingkan_regresi.py     regresi 9 faktor di protokol yang sama
src/buat_manifest.py    sha256 berkas + angka utama -> manifest.json
outputs/model/model_bag.joblib     artefak siap pakai: bag 6 model (6 MB)
outputs/model/cv.json   hasil CV populasi 306: MAE+CI95, R², akurasi kategori + QWK, per bin,
                        tabel silang + rincian per kategori
outputs/model/per_bin.csv
outputs/model/produksi.json        spesifikasi artefak produksi + ambang kategori
outputs/model/perbandingan_regresi.json   16 varian regresi 9 faktor + baris yang dipakai
outputs/model/manifest.json        sha256 tiap berkas + angka utama (deterministik)
data/split_v1.json      sidik jari pembagian data (sha 1d5f63efb7d61488), bukan dasar penilaian
docs/codebook.md        definisi variabel, aturan pembersihan, asumsi yang perlu dikonfirmasi
docs/Laporan-Model-Kecemasan-Remaja.docx   laporan teknis (15 halaman) untuk pembaca non-teknis
requirements.txt        versi paket dipatok (termasuk pyreadstat untuk membaca .sav)
```

Tidak ikut repo — berisi jawaban per siswa atau data mentah: `data/raw/`,
`data/features.parquet`, `outputs/model/oof.csv`, `outputs/model/prediksi*.csv`. Karena data
mentah dan matriks fiturnya tidak ikut, `--mode cv`, `--mode test`, dan `--mode final` hanya
bisa dijalankan oleh yang punya `data/raw/INPUT DATA  (1).sav`; dari situ
`python src/build_features.py` membangun ulang matriks fiturnya. `--mode predict` tetap jalan
dari hasil clone karena artefak model sudah ikut repo. sha256 berkas privat dicatat di
`manifest.json`, jadi pemilik data bisa mencocokkan berkas yang dipakai di sini.

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python src/build_features.py              # .sav -> data/features.parquet (butuh data mentah)
python src/build_features.py --check      # buktikan hasilnya identik dengan berkas yang dipakai
python src/train.py --mode check-split   # verifikasi split terkunci
python src/train.py --mode self-test     # uji fungsi murni
python src/train.py --mode cv            # CV 5x6 -> outputs/model/cv.json
python src/train.py --mode final         # bag 6 model -> model_bag.joblib
python src/train.py --mode predict --input siswa_baru.csv   # skor + label + alasan
python src/bandingkan_regresi.py         # regresi 9 faktor, protokol sama -> JSON
python src/buat_manifest.py              # sha256 + angka utama -> manifest.json
python src/buat_manifest.py --check      # cocokkan berkas dengan manifest tersimpan
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

## Peta verifikasi

Setiap angka yang dilaporkan bisa ditelusuri ke satu berkas. "Privat" = butuh data mentah yang
tidak ikut repo.

| Klaim | Bukti | Cara periksa |
|---|---|---|
| 306 siswa; 76 kolom input; target = rata-rata 20 item | `src/build_features.py` (privat: `features.parquet`) | `python src/build_features.py --check` |
| MAE CV 0,2165 [0,196; 0,237]; R² 0,6305; 69,3% kategori; QWK 0,734 | `outputs/model/cv.json` | baca JSON; hitung ulang `--mode cv` (privat) |
| MAE in-sample bag 0,1288; R² 0,8502 | `outputs/model/produksi.json` | `--mode final` (privat) |
| Galat per bin; rincian per kategori | `outputs/model/per_bin.csv`, `cv.json → kategori` | baca berkas (privat untuk hitung ulang) |
| Berkas split tidak berubah | `data/split_v1.json` + `SHA_V1` di `train.py` | `--mode check-split` (privat) |
| 6 model produksi #8/#11/#17/#20/#24/#26 | `produksi.json → pemilihan_model`, `train.py → PEMILIHAN_BAG` | baca dua berkas |
| Regresi 9 faktor: MAE CV 0,2469, R² 0,534 | `outputs/model/perbandingan_regresi.json` | `python src/bandingkan_regresi.py` (privat) |
| Angka ↔ berkas yang menghasilkannya | `outputs/model/manifest.json` | `python src/buat_manifest.py --check` |

Tanpa data mentah, yang bisa diperiksa langsung dari hasil clone: seluruh angka di `cv.json`,
`produksi.json`, `perbandingan_regresi.json`, `manifest.json`, plus `--mode
self-test` dan `--mode predict` pada CSV sendiri. Yang tidak bisa: menghitung ulang angkanya.

## Catatan pembacaan

- 0,2165 adalah estimasi **populasi**: himpunan evaluasi CV = seluruh 306 siswa.
- Tidak ada angka uji di repo ini maupun di laporan. Berkas split tetap disimpan untuk
  memeriksa integritas data, tetapi tidak dipakai menilai model.
- Ada **dua baris duplikat** (row_id 53 dan 55, identik di seluruh 76 kolom input dan nilai
  y-nya) yang sengaja tidak dihapus agar N tetap 306. Pada 10 dari 30 model CV, satu kembar
  ada di data latih dan satunya di data uji, sehingga MAE CV sedikit optimistis: tanpa kedua
  baris itu MAE menjadi **0,2179** (batas atas, bukan hasil retrain).
- Urutan kolom pada berkas input **tidak** berpengaruh — kolom dipilih berdasarkan nama.
  Yang berpengaruh: nama kolom, skala nilai, dan level kategori. Level kategori di luar data
  latih sekarang ditolak dengan pesan, tidak lagi berubah menjadi dummy nol secara senyap.
- Tabel per bin (lebar 0,25) dan tabel kategori (tiga tingkat) memakai batas yang tidak
  berimpit: `1,70` sudah **Rendah** padahal masih **bin 3**, dan `2,20` sudah **Sedang**
  padahal masih **bin 5**. Jangan membaca tabel bin sebagai tabel kategori.
- Bin 10 hanya 3 siswa (bin 9: 6, bin 8: 9). MAE per bin untuk bin dengan n ≤ 5 dilaporkan
  tetapi tidak boleh diranking.
- Kategori adalah pembagian skor prediksi pada ambang tetap, bukan model klasifikasi
  tersendiri. Ambang dipilih untuk QWK, bukan untuk keputusan penyaringan; untuk keperluan
  triase, "Tinggi" (88 siswa, 29%) adalah penanda awal, bukan diagnosis.

# Codebook — Data dan Fitur Model Risiko Kecemasan

Disusun dari data mentah 306 siswa SMK, Kec. Kresek, Kab. Tangerang (survei cross-sectional).

## Target

| Variabel | Definisi | Rentang | Distribusi |
|---|---|---|---|
| `y_kontinu` | Rata-rata 20 item kecemasan/stres (skala 1=Tidak pernah … 4=Sangat sering) | 1.00–4.00 | 1.00–3.50 (mean 1.97, SD 0.47) |
| `y_biner` | 1 bila `y_kontinu` ≥ 2.5 (proxy kategori *cemas tinggi*) | 0/1 | 1 = 43 siswa (14.1%) |

> **Catatan ketidakseimbangan kelas:** hanya 43/306 siswa (14.1%) berada di kategori positif. Model biner wajib memakai `scale_pos_weight` atau `class_weight`, dan dievaluasi dengan AUC + Brier score (bukan accuracy).

## 18 Fitur Input

| # | Fitur | Sumber item | Skala | Rentang data |
|---|---|---|---|---|
| 1 | `f_akademik` | Faktor tekanan akademik — AKADEMIK1, AKADEMIK2, AKADEMIK3 | 1–5 | 1.00–4.67 (mean 2.46, SD 0.89) |
| 2 | `f_keluarga` | Faktor tekanan keluarga — KELUARGA1, KELUARGA2, KELUARGA3 | 1–5 | 1.00–5.00 (mean 2.31, SD 1.08) |
| 3 | `f_sosial` | Faktor tekanan sosial (SOSIAL3 sudah di-reverse) — SOSIAL1, SOSIAL2, SOSIAL3 | 1–5 | 1.00–5.00 (mean 2.74, SD 0.72) |
| 4 | `f_ekonomi` | Faktor tekanan ekonomi — EKONOMI1, EKONOMI2, EKONOMI3 | 1–5 | 1.00–5.00 (mean 2.03, SD 1.00) |
| 5 | `f_digital` | Faktor tekanan digital — DIGITAL1, DIGITAL2, DIGITAL3 | 1–5 | 1.00–4.67 (mean 1.77, SD 0.79) |
| 6 | `f_masadepan` | Faktor tekanan masa depan — MASADEPAN1, MASADEPAN2, MASADEPAN3 | 1–5 | 1.00–4.67 (mean 2.37, SD 0.84) |
| 7 | `f_perilaku_sehat` | Perilaku sehat (konstruk inti penelitian) — PERILAKU1–PERILAKU10 (10 item) | 1–5 | 1.30–4.70 (mean 2.96, SD 0.57) |
| 8 | `f_duk_keluarga` | Dukungan lingkungan — keluarga — LINGKUNGAN1–4 | 1–7 | 1.00–7.00 (mean 4.81, SD 1.26) |
| 9 | `f_duk_teman` | Dukungan lingkungan — teman — LINGKUNGAN5–8 | 1–7 | 1.00–7.00 (mean 4.33, SD 1.22) |
| 10 | `f_duk_orang_dekat` | Dukungan lingkungan — orang terdekat — LINGKUNGAN9–12 | 1–7 | 1.00–7.00 (mean 4.96, SD 1.38) |
| 11 | `f_coping_adaptif` | Coping adaptif (fokus masalah, religius, cari bantuan) — COPING1–12 | 1–5 | 1.58–4.67 (mean 3.34, SD 0.57) |
| 12 | `f_coping_ekspresi` | Coping ekspresi emosi — COPING13–15 | 1–5 | 1.00–5.00 (mean 2.59, SD 1.00) |
| 13 | `f_coping_negatif` | Coping negatif (menyendiri, self-blame, helpless) — COPING16–18 | 1–5 | 1.00–5.00 (mean 2.72, SD 0.95) |
| 14 | `d_umur` | Umur siswa — Umur | 14–17 | 4 kategori |
| 15 | `d_jenis_kelamin` | Jenis kelamin — JK | L / P | 2 kategori |
| 16 | `d_jurusan` | Jurusan — Jurusan | 5 kategori | 5 kategori |
| 17 | `d_tinggal_grup` | Tinggal dengan siapa (dipadatkan 11 → 5 grup) — TINGGAL → TINGGAL_RECODE | 1–5 | 5 kategori |
| 18 | `d_status_ortu` | Status orang tua (dipadatkan 5 → 4 kode) — ORANGTUA → ORANGTUA_RECODE | 1–4 | 4 kategori |

## 88 Fitur Model

Model akhir memakai 88 kolom, semuanya diturunkan dari data mentah yang sama:

| blok | jumlah | isi |
|---|---|---|
| skor faktor | 13 | 6 faktor tekanan, perilaku sehat, 3 dukungan lingkungan, 3 coping |
| demografi numerik | 1 | `d_umur` |
| item faktor tekanan | 18 | AKADEMIK1–3, KELUARGA1–3, SOSIAL1–3, EKONOMI1–3, DIGITAL1–3, MASADEPAN1–3 |
| item coping | 18 | COPING1–18 |
| item perilaku sehat | 10 | PERILAKU1–10 |
| item dukungan lingkungan | 12 | LINGKUNGAN1–12 |
| demografi one-hot | 16 | jenis kelamin, jurusan, tinggal, status orang tua |

Urutan kolom pada matriks latih bersifat tetap; model memakai urutan itu apa adanya.

## Aturan Cleaning (Tahap 1)

| Aturan | Perlakuan |
|---|---|
| Item kecemasan 13+14 dan 19+20 | Kalimat terbelah jadi 2 kolom; digabung (rata-rata nilai sah), kolom pasangan (14, 20) dibuang — keduanya berisi 0 di semua baris |
| `KECEMASANSTRESS12` = 5 | Di luar skala 1–4 → imputasi median kolom |
| Item terbalik (SOSIAL3) | Reverse-score: nilai_baru = 6 − nilai |
| `VAR00001` | Kolom kosong → dibuang |
| `KECEMSANSTRESS9` (salah ketik) | Nama kolom item 9 → `KECEMASANSTRESS9` |

- **Audit arah item (hasil verifikasi).** `REVERSE_ITEMS=[SOSIAL3]` sudah lengkap: semua item bernegasi lain (KECEMASANSTRESS2/9/16/17/22, MASADEPAN3, COPING16–18) sengaja TIDAK di-reverse karena tetap searah konstruk/subskalanya (COPING16–18 = subskala `coping_negatif`, skor tinggi = makin negatif).
- **Sensitivitas item 12 (nilai 5).** Nilai invalid diisi median kolom; alternatif isi 4 teruji TIDAK mengubah label `y_biner` siswa mana pun.

## Kategori yang Dilaporkan (3 Tingkat)

Ambang `1.70` dan `2.20` pada `y_kontinu`; keduanya kira-kira rata-rata sampel (1,967) dikurangi dan ditambah 0,5 simpangan baku (0,467).

| kategori | batas | n |
|---|---|---|
| Rendah | ≤ 1,70 | 103 |
| Sedang | 1,70 – 2,20 | 115 |
| Tinggi | > 2,20 | 88 |

- Ambang dipilih dari grid dua parameter pada prediksi out-of-fold dengan kriteria **QWK**. Memilih ambang demi akurasi menghasilkan skema yang menumpuk 238 siswa di satu kelas (akurasi 0,83 tetapi QWK 0,56), jadi akurasi tidak dipakai sebagai kriteria.
- Uji bersarang (ambang dipilih di lipatan latih, dinilai di lipatan uji, 30 lipatan): **QWK 0,689, akurasi 66,0%**. Angka pada populasi (0,734 / 69,3%) karena itu sedikit optimistis, sebagaimana setiap ambang yang dipilih pada data yang sama.
- Skema empat tingkat dokumen sebelumnya (115/153/35/3 dengan ambang 1,75/2,50/3,25) tidak dipakai: kelas teratas hanya 3 siswa dengan recall 1 dari 3 sehingga tidak dapat dipertanggungjawabkan. Angka itu tetap dicatat sebagai pembanding.
- `y_biner` di `features.parquet` tetap memakai 2,5 dan **bukan** kategori yang dilaporkan; ia warisan definisi biner lama dan hanya disimpan untuk kompatibilitas.

## Asumsi yang Perlu Dikonfirmasi Dosen

- **Label status orang tua.** Kategori 1 berlabel *"kedua orang tua meninggal dunia"* tapi 226 dari 240 siswa kategori tersebut tinggal bersama ayah dan ibu → label kemungkinan salah ketik. Asumsi kerja: kategori 1 = **keluarga utuh**. Kategori 2 (9 siswa, pola mencampur) dan kategori 5 (4 siswa, tanpa label) digabung ke **lainnya**. Sisa kategori 3 & 4 dipakai sesuai label aslinya. 3 versi coding disiapkan untuk sensitivity analysis, namun BELUM dijalankan (fase robustness terpisah).
- **Ambang `y_biner`.** 2,5 dipilih sebagai titik tengah antara pilihan "kadang-kadang" (2) dan "sering" (3) — siswa dengan rata-rata skor di atasnya masuk label biner cemas tinggi (lihat bagian Kategori yang Dilaporkan untuk kategori tiga tingkat yang dipakai model); ambang alternatif 2,0 dan 2,3 menghasilkan 42,2% dan 25,8% positif — dipakai sebagai bahan diskusi bila dosen ingin prevalensi lebih tinggi.
- **Temuan penting — `f_sosial` berbeda dari analisis sebelumnya.** Rata-rata blok sosial TANPA reverse-scoring adalah **2,15** (angka yang dipakai di analisis deskriptif & regresi lama), sedangkan pipeline kita memakai mean **2,74** setelah `SOSIAL3` di-reverse. Item itu berbunyi *"Saya merasa TIDAK tertekan ketika mengalami masalah…"* — arahnya berlawanan, sehingga skor sebelumnya sempat menyamakan "sering tidak tertekan" dengan "tinggi tekanan sosial". Karena faktor sosial menjadi prediktor terkuat di regresi lama (β=0,241), temuan itu perlu dihitung ulang. Pipeline ini sengaja mengikuti aturan `config.REVERSE_ITEMS` (reverse = benar).
- **Baris duplikat.** Dua baris identik di seluruh kolom tidak dihapus (N tetap 306) agar selaras dengan analisis deskriptif sebelumnya; ditandai saat audit data: row_id 53 dan 55, y = 1,65. Implikasi evaluasi (terukur, bukan perkiraan): pasangan kembar terpisah ke data latih dan data uji pada 10 dari 30 model CV, dan tanpa kedua baris itu MAE CV menjadi 0,2179 bukan 0,2165 — optimisme **0,0013 (0,6%)**.

## Pemakaian Etis

Output model bersifat **agregat dan eksploratif** untuk policy promotif-preventif; bukan alat diagnosis klinis dan bukan label per individu.

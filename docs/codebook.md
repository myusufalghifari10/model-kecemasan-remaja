# Codebook — Data dan Fitur Model Risiko Kecemasan

Disusun dari data mentah 306 siswa SMK, Kec. Kresek, Kab. Tangerang (survei cross-sectional).

## Target

| Variabel | Definisi | Rentang | Distribusi |
|---|---|---|---|
| `y_kontinu` | Rata-rata 20 item kecemasan/stres (skala 1=Tidak pernah … 4=Sangat sering) | 1.00–4.00 | 1.00–3.50 (mean 1.97, SD 0.47) |
| `y_biner` | 1 bila `y_kontinu` ≥ 2.5 (proxy kategori *cemas tinggi*) | 0/1 | 1 = 43 siswa (14.1%) |

> **Dua konvensi pada angka 2,5.** `y_biner` memakai **≥ 2,5** (43 siswa, 14,1%). Angka sensitivitas/spesifisitas di laporan memakai **> 2,50** (38 siswa) karena kalimatnya berbunyi "di atas 2,50". Bedanya 5 siswa yang nilainya tepat 2,50: dengan ≥ sensitivitas 51,2% dan spesifisitas 93,9%, dengan > sensitivitas 50,0% dan spesifisitas 92,9%. Jangan mencampur keduanya.

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
- **Peringatan pada SOSIAL3 (temuan audit, perlu keputusan pembimbing).** Pembalikan `6 − x` menghasilkan `f_sosial` yang korelasinya dengan skor kecemasan **+0,381**, sedangkan versi tanpa pembalikan **+0,536**; jadi pembalikan tidak membalik arah faktor, hanya melemahkan korelasinya. Nilai mentah item itu sendiri berkorelasi **+0,209** dengan skor kecemasan, dan model memberi sumbangan lebih tinggi pada siswa yang menjawab "selalu tidak tertekan" (sumbangan +0,098 pada nilai tersimpan 1, yaitu mentah 5) daripada yang menjawab "tidak pernah tidak tertekan" (−0,018 pada nilai tersimpan 5, mentah 1). Artinya pada data ini item tersebut berperilaku seolah **tidak** perlu dibalik. Pipeline tetap mengikuti `config.REVERSE_ITEMS` agar konsisten dan dapat direproduksi, tetapi arah item ini harus diputuskan bersama pembimbing sebelum hasilnya ditafsirkan secara substantif.
- **Sensitivitas item 12 (nilai 5).** Nilai invalid diisi median kolom; alternatif isi 4 teruji TIDAK mengubah label `y_biner` siswa mana pun.

## Kategori yang Dilaporkan (3 Tingkat)

Ambang `1.70` dan `2.20` pada `y_kontinu`; keduanya kira-kira rata-rata sampel (1,967) dikurangi dan ditambah 0,5 simpangan baku (0,467).

| kategori | batas | n |
|---|---|---|
| Rendah | ≤ 1,70 | 103 |
| Sedang | 1,70 – 2,20 | 115 |
| Tinggi | > 2,20 | 88 |

- Ambang dipilih dari grid dua parameter pada prediksi out-of-fold dengan kriteria **QWK**. Memilih ambang demi akurasi menghasilkan skema yang menumpuk hampir semua siswa di satu kelas (264 dari 306 siswa aktual, dan 281 dari 306 prediksi menunjuk ke kelas itu) dengan akurasi 88,6% tetapi QWK 0,54, jadi akurasi tidak dipakai sebagai kriteria. Kedua angka dihitung ulang oleh `python src/uji_ambang.py`.
- Uji bersarang penuh (ambang dipilih lewat CV 5 lipatan di dalam lipatan latih, model dilatih ulang, lalu dinilai di lipatan uji, 30 lipatan): **QWK gabungan 306 siswa 0,725 dan akurasi 68,3%**; rata-rata per lipatan 0,679 dan 65,3%. Pembanding setaranya bukan 0,734 (gabungan enam model), melainkan satu model per lipatan dengan ambang tetap pada lipatan yang sama: **0,739**, karena uji bersarang juga memakai satu model per lipatan. Jadi harga pemilihan ambang yang tidak melihat siswa uji sekitar 0,013 QWK.
  Bedanya dua estimand itu: `gabungan_306` mengunci kategori aktual di 1,70/2,20 (sebanding dengan angka 0,734), sedangkan `rata_rata_per_lipatan` mengategorikan nilai aktual di lipatan uji dengan ambang lipatan itu sendiri. Yang dibandingkan dengan 0,734 adalah angka gabungan.
- Skema empat tingkat dokumen sebelumnya (115/153/35/3 dengan ambang 1,75/2,50/3,25) tidak dipakai: kelas teratas hanya 3 siswa dengan recall 1 dari 3 sehingga tidak dapat dipertanggungjawabkan. Angka itu tetap dicatat sebagai pembanding.
- `y_biner` di `features.parquet` tetap memakai 2,5 dan **bukan** kategori yang dilaporkan; ia warisan definisi biner lama dan hanya disimpan untuk kompatibilitas.

## Asumsi yang Perlu Dikonfirmasi Dosen

- **Label status orang tua.** Kategori 1 berlabel *"kedua orang tua meninggal dunia"* tapi 226 dari 240 siswa kategori tersebut tinggal bersama ayah dan ibu → label kemungkinan salah ketik. Asumsi kerja: kategori 1 = **keluarga utuh**. Kategori 2 (9 siswa, pola mencampur) dan kategori 5 (4 siswa, tanpa label) digabung ke **lainnya**. Sisa kategori 3 & 4 dipakai sesuai label aslinya. 3 versi coding disiapkan untuk sensitivity analysis, namun BELUM dijalankan (fase robustness terpisah).
- **Ambang `y_biner`.** 2,5 dipilih sebagai titik tengah antara pilihan "kadang-kadang" (2) dan "sering" (3) — siswa dengan rata-rata skor di atasnya masuk label biner cemas tinggi (lihat bagian Kategori yang Dilaporkan untuk kategori tiga tingkat yang dipakai model); ambang alternatif 2,0 dan 2,3 menghasilkan 42,2% dan 25,8% positif — dipakai sebagai bahan diskusi bila dosen ingin prevalensi lebih tinggi.
- **Temuan penting — `f_sosial` berbeda dari analisis sebelumnya.** Rata-rata blok sosial TANPA reverse-scoring adalah **2,15** (angka yang dipakai di analisis deskriptif & regresi lama), sedangkan pipeline kita memakai mean **2,74** setelah `SOSIAL3` di-reverse; korelasinya dengan skor kecemasan berubah dari **+0,536** menjadi **+0,381** (tetap positif, hanya lebih lemah). Item itu berbunyi *"Saya merasa TIDAK tertekan ketika mengalami masalah…"* — arahnya berlawanan, sehingga skor sebelumnya sempat menyamakan "sering tidak tertekan" dengan "tinggi tekanan sosial". Karena faktor sosial menjadi prediktor terkuat di regresi lama (β=0,241), koefisien sosial di regresi pembanding tidak boleh dibandingkan langsung dengan angka lama. Pipeline ini sengaja mengikuti aturan `config.REVERSE_ITEMS`, tetapi lihat peringatan pada SOSIAL3 di atas: data ini berperilaku sebaliknya, jadi arah item menunggu keputusan pembimbing.
- **Baris duplikat.** Dua baris identik di seluruh kolom tidak dihapus (N tetap 306) agar selaras dengan analisis deskriptif sebelumnya; ditandai saat audit data: row_id 53 dan 55, y = 1,65. Implikasi evaluasi (terukur, bukan perkiraan): pasangan kembar terpisah ke data latih dan data uji pada 10 dari 30 model CV, dan tanpa kedua baris itu MAE CV menjadi 0,2179 bukan 0,2165 — optimisme **0,0013 (0,6%)**.

## Reliabilitas Instrumen

Angka reliabilitas tidak disimpan di berkas artefak karena dihitung dari data mentah (privat). Cara menghitungnya, untuk pemeriksaan ulang dengan `.sav` asli:

- **Cronbach alpha 20 item kecemasan = 0,892.** Ini reliabilitas yang dipakai di laporan §2.3 dan §5.
- **Standard error of measurement (SEM) = 0,153**, yaitu SD skor (0,467) dikali akar(1 - alpha). Setara dengan sekitar sepertiga simpangan baku.
- **Jarak ke batas kategori.** Dengan SEM 0,153, siswa yang nilainya dalam ±0,153 dari batas 1,70 atau 2,20 (satu SEM) sebanyak **160 dari 306**; dengan ±0,31 (dua SEM, selang 95%) sebanyak **240 dari 306**. Ini alasan label di dekat batas tidak stabil (§8 laporan). Angka 140 yang muncul bila dipakai ±0,15 bukan SEM (0,153), jadi tidak dipakai.
- **Blok sosial (3 item).** Alpha 0,259 setelah SOSIAL3 dibalik dan 0,491 dengan nilai mentah; korelasi item-total SOSIAL3 -0,121 setelah dibalik dan +0,121 mentah. Karena itu tafsir faktor sosial ditahan sampai kunci item diputuskan.
- **Uji profil 3 x 4.** Kolom Sangat Tinggi hanya berisi 3 siswa, sehingga frekuensi harapan di kolom itu sekitar 1 per sel (1,11 / 0,86 / 1,03 untuk tiga profil). Chi-square tidak sah tanpa penggabungan kelas atau uji eksak.

## Pemakaian Etis

Output model bersifat **agregat dan eksploratif** untuk policy promotif-preventif; bukan alat diagnosis klinis dan bukan label per individu.

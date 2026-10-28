"""Bangun matriks fitur dari data mentah SPSS — tahap 1 pipeline.

Alur (semua aturan di `docs/codebook.md`, bagian "Aturan Cleaning"):

1. baca `data/raw/INPUT DATA  (1).sav` (306 baris, 87 kolom);
2. perbaiki nama kolom yang salah ketik (`KECEMSANSTRESS9` -> item 9);
3. gabung item kecemasan yang kalimatnya terbelah jadi dua kolom: pasangan
   (13, 14) dan (19, 20) — kolom kedua selalu 0, jadi gabungan = kolom pertama;
4. buang kolom pasangan yang kosong, sehingga target = rata-rata 20 item;
5. nilai di luar skala 1-4 (`KECEMASANSTRESS12` = 5) diisi median kolom sah;
6. `SOSIAL3` di-reverse (6 - nilai) sebelum dirata-rata menjadi `f_sosial`;
7. `y_biner` = 1 bila `y_kontinu` >= 2.5 (43 siswa).

Keluaran: parquet berisi `row_id` (0..305), 76 kolom masukan model, dan dua kolom
target. `w_style` / `longstring_max` / `flag_straightliner` dari berkas lama TIDAK
dibangun lagi: ketiganya tidak dipakai model, dan aturan pembuatannya tidak
terdokumentasi di repo mana pun.

Pakai:
  python src/build_features.py                      # tulis data/features.parquet
  python src/build_features.py --output /tmp/f.parquet
  python src/build_features.py --check              # bandingkan hasil dengan berkas yang ada
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pyreadstat

import config as C

# Item kecemasan 1-22. Kolom 9 salah ketik di berkas sumber.
ITEM_CE_MENTAH = [f"KECEMASANSTRESS{i}" if i != 9 else "KECEMSANSTRESS9" for i in range(1, 23)]
ITEM_CE_NAMA = [c.replace("KECEMSANSTRESS", "KECEMASANSTRESS") for c in ITEM_CE_MENTAH]
PASANGAN_KOSONG = ["KECEMASANSTRESS14", "KECEMASANSTRESS20"]   # selalu 0,0
ITEM_CE_SAH = [c for c in ITEM_CE_NAMA if c not in PASANGAN_KOSONG]   # 20 item
SKALA_CE = (1, 4)

# TINGGAL 11 nilai -> 5 grup; ORANGTUA 5 kode -> 4 kode.
# Peta ini dibaca dari sebaran data (crosstab nilai mentah vs kolom turunan).
PETA_TINGGAL = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 9: 4, 10: 5, 11: 5}
PETA_ORANGTUA = {1: 1, 2: 4, 3: 2, 4: 3, 5: 4}

KOLOM_ITEM = ([c for blok in C.FACTOR_ITEMS.values() for c in blok]
              + [f"COPING{i}" for i in range(1, 19)]
              + C.PERILAKU_ITEMS + C.LINGKUNGAN_ITEMS)
KOLOM_DEMOGRAFI = ["Umur", "JK", "Jurusan", "TINGGAL", "ORANGTUA"]

URUTAN_KOLOM = (
    ["row_id"] + C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + KOLOM_ITEM
    + [C.TARGET_CONT, "y_biner"]
)


def muat_mentah() -> pd.DataFrame:
    if not C.DATA_RAW.is_file():
        _fail(f"tidak ada data mentah: {C.DATA_RAW} (arsip data, tidak ikut repo)")
    df, _ = pyreadstat.read_sav(str(C.DATA_RAW))
    if df.shape[0] != 306:
        _fail(f"data mentah berisi {df.shape[0]} baris, diharapkan 306")
    hilang = [c for c in ITEM_CE_MENTAH if c not in df.columns]
    if hilang:
        _fail(f"kolom item kecemasan tidak ada di berkas mentah: {hilang[:3]}")
    return df


def bersihkan_kecemasan(raw: pd.DataFrame) -> pd.DataFrame:
    """20 item sah: pasangan kosong digabung lalu dibuang, nilai di luar skala -> median."""
    ce = raw[ITEM_CE_MENTAH].copy()
    ce.columns = ITEM_CE_NAMA
    for kosong, induk in (("KECEMASANSTRESS14", "KECEMASANSTRESS13"),
                          ("KECEMASANSTRESS20", "KECEMASANSTRESS19")):
        dua = ce[[induk, kosong]].replace(0.0, np.nan)      # 0 = kolom tidak terpakai
        ce[induk] = dua.mean(axis=1)
    ce = ce[ITEM_CE_SAH]
    lo, hi = SKALA_CE
    for kol in ce.columns:
        salah = (ce[kol] < lo) | (ce[kol] > hi)
        if salah.any():
            ce.loc[salah, kol] = ce.loc[~salah, kol].median()
    if ce.isna().any().any():
        _fail("masih ada nilai kosong pada item kecemasan setelah pembersihan")
    return ce


def skor_faktor(item: pd.DataFrame) -> pd.DataFrame:
    """13 skor faktor dari item yang sudah dibersihkan (SOSIAL3 di sini sudah reversed)."""
    f = pd.DataFrame(index=item.index)
    for nama, kolom in C.FACTOR_ITEMS.items():
        f[f"f_{nama}"] = item[kolom].mean(axis=1)
    f["f_perilaku_sehat"] = item[C.PERILAKU_ITEMS].mean(axis=1)
    f["f_duk_keluarga"] = item[[f"LINGKUNGAN{i}" for i in range(1, 5)]].mean(axis=1)
    f["f_duk_teman"] = item[[f"LINGKUNGAN{i}" for i in range(5, 9)]].mean(axis=1)
    f["f_duk_orang_dekat"] = item[[f"LINGKUNGAN{i}" for i in range(9, 13)]].mean(axis=1)
    f["f_coping_adaptif"] = item[[f"COPING{i}" for i in range(1, 13)]].mean(axis=1)
    f["f_coping_ekspresi"] = item[[f"COPING{i}" for i in range(13, 16)]].mean(axis=1)
    f["f_coping_negatif"] = item[[f"COPING{i}" for i in range(16, 19)]].mean(axis=1)
    f["d_umur"] = item["Umur"].astype(int)
    f["d_jenis_kelamin"] = item["JK"]
    f["d_jurusan"] = item["Jurusan"]
    f["d_tinggal_grup"] = item["TINGGAL"].map(PETA_TINGGAL).astype(int)
    f["d_status_ortu"] = item["ORANGTUA"].map(PETA_ORANGTUA).astype(int)
    if f[C.NUMERIC_FEATURES].isna().any().any() or f[C.CATEGORICAL_FEATURES].isna().any().any():
        _fail("peta TINGGAL/ORANGTUA tidak mencakup semua nilai di data mentah")
    return f


def bangun(raw: pd.DataFrame) -> pd.DataFrame:
    ce = bersihkan_kecemasan(raw)
    sumber = raw[KOLOM_ITEM + KOLOM_DEMOGRAFI].copy()
    sumber["SOSIAL3"] = 6 - sumber["SOSIAL3"]      # berkas fitur menyimpan nilai yang sudah di-reverse
    fitur = skor_faktor(sumber)
    y = ce.mean(axis=1)
    df = pd.concat([fitur, sumber[KOLOM_ITEM]], axis=1)
    df.insert(0, "row_id", np.arange(len(df)))
    df[C.TARGET_CONT] = y
    df["y_biner"] = (y >= 2.5).astype(int)
    return df[URUTAN_KOLOM]


def periksa(path_hasil: str) -> int:
    """Bandingkan hasil bangun ulang dengan matriks fitur yang dipakai model."""
    baru = pd.read_parquet(path_hasil)
    lama = pd.read_parquet(C.DATA_FEATURES)
    gagal = 0
    for kol in URUTAN_KOLOM:
        a, b = baru[kol], lama[kol]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            selisih = float(np.abs(a.to_numpy(float) - b.to_numpy(float)).max())
        else:
            selisih = 0.0 if (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all() else 1.0
        status = "OK " if selisih == 0.0 else "BEDA"
        if selisih != 0.0:
            gagal += 1
        print(f"  {status} {kol:<22} selisih maks {selisih:.3e}")
    print(f"[build_features] {len(URUTAN_KOLOM) - gagal}/{len(URUTAN_KOLOM)} kolom identik dengan "
          f"{C.DATA_FEATURES.name}")
    return 0 if gagal == 0 else 1


def _fail(pesan: str) -> None:
    print(f"[build_features] {pesan}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    p = argparse.ArgumentParser(description="Bangun matriks fitur dari data mentah SPSS.")
    p.add_argument("--output", default=str(C.DATA_FEATURES))
    p.add_argument("--check", action="store_true",
                   help="bandingkan hasil dengan matriks fitur yang sudah ada")
    a = p.parse_args()

    raw = muat_mentah()
    df = bangun(raw)
    tujuan = a.output
    df.to_parquet(tujuan, index=False)
    print(f"[build_features] {df.shape[0]} baris x {df.shape[1]} kolom -> {tujuan}")
    print(f"[build_features] y_kontinu {df[C.TARGET_CONT].min():.2f}-{df[C.TARGET_CONT].max():.2f} "
          f"(rata-rata {df[C.TARGET_CONT].mean():.3f}, SD {df[C.TARGET_CONT].std(ddof=1):.3f}) | "
          f"y_biner positif {int(df['y_biner'].sum())}")
    if a.check:
        return periksa(tujuan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

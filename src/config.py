"""Konfigurasi pusat: satu-satunya sumber kebenaran untuk nama kolom dan path.

Model: PLS1 (tulang punggung linier) + XGBoost pada residual, dengan bobot per bin
skor kecemasan. Target `y_kontinu` adalah rata-rata 20 item kecemasan (skala 1-4).
"""
from pathlib import Path

# ---------- Path ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "INPUT DATA  (1).sav"
DATA_FEATURES = ROOT / "data" / "features.parquet"      # 306 x 89 (88 fitur + target)
DATA_SPLIT = ROOT / "data" / "split_v1.json"            # 238 latih / 68 uji (terkunci)
CODEBOOK = ROOT / "docs" / "codebook.md"

# ---------- Reproducibility ----------
RANDOM_SEED = 42
BASE_XGB = {"random_state": RANDOM_SEED, "n_jobs": -1, "verbosity": 0}

# ---------- Inventaris item (nama kolom persis seperti di data mentah) ----------
# Item faktor tekanan psikososial (skala 1-5)
FACTOR_ITEMS = {
    "akademik": ["AKADEMIK1", "AKADEMIK2", "AKADEMIK3"],
    "keluarga": ["KELUARGA1", "KELUARGA2", "KELUARGA3"],
    "sosial": ["SOSIAL1", "SOSIAL2", "SOSIAL3"],
    "ekonomi": ["EKONOMI1", "EKONOMI2", "EKONOMI3"],
    "digital": ["DIGITAL1", "DIGITAL2", "DIGITAL3"],
    "masadepan": ["MASADEPAN1", "MASADEPAN2", "MASADEPAN3"],
}

# Strategi coping (skala 1-5)
COPING_ITEMS = {f"COPING{i}" for i in range(1, 19)}

# Perilaku sehat (skala 1-5)
PERILAKU_ITEMS = [f"PERILAKU{i}" for i in range(1, 11)]

# Dukungan lingkungan / MSPSS (skala 1-7)
LINGKUNGAN_ITEMS = [f"LINGKUNGAN{i}" for i in range(1, 13)]

# ---------- Target ----------
TARGET_CONT = "y_kontinu"      # rata-rata 20 item kecemasan valid (1-4)

# ---------- Skor faktor + demografi (dipakai untuk model pembanding) ----------
NUMERIC_FEATURES = [
    "f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital", "f_masadepan",
    "f_perilaku_sehat",
    "f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat",
    "f_coping_adaptif", "f_coping_ekspresi", "f_coping_negatif",
    "d_umur",
]
CATEGORICAL_FEATURES = ["d_jenis_kelamin", "d_jurusan", "d_tinggal_grup", "d_status_ortu"]

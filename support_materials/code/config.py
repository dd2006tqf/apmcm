"""Global configuration: paths, seeds and every assumption switch of MODELING_REPORT.md §1 (A1-A16).

Observed quantities are read from data; the constants below are the *assumed* parameters
(each one is scanned in a sensitivity analysis, see problem3/problem5).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("FLOOD_DATA_DIR", ROOT / "user_data" / "a题数据汇总"))
_OFFICIAL_ROOT = DATA_DIR / "official" if (DATA_DIR / "official").is_dir() else DATA_DIR
_PUBLIC_ROOT = DATA_DIR / "collected" / "公开数据" if (DATA_DIR / "collected" / "公开数据").is_dir() else DATA_DIR / "公开数据"
OFFICIAL_LABEL_DIR = _OFFICIAL_ROOT / "Problem A Dataset 1" / "Dataset" / "label information"
OFFICIAL_SAR_DIR = _OFFICIAL_ROOT / "Problem A Dataset 1" / "Dataset" / "post-disaster SAR images"
OFFICIAL_OPT_DIR = _OFFICIAL_ROOT / "Problem A Dataset 2" / "Dataset" / "pre-disaster optical images"
PUB = _PUBLIC_ROOT

OUT = ROOT / "output"
FIG = ROOT / "figures"
CACHE = ROOT / "_tmp" / "cache"          # large regenerable caches (features, probability maps)
for _d in (OUT, FIG, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_WORKERS = max(1, int(os.environ.get("MODEX_MAX_WORKERS") or 4))

# ---------------- official tiles (Q1/Q2) ----------------
TILE = 256
SCALE_DEG = 8.333e-5
LABEL_SEMANTICS = "new"        # A1: labels = new inundation (re-tested by P1-C7)
BLOCK = 5                      # A2: 5x5-tile spatial blocks
N_FOLDS = 5
BUFFER_TILES = 1
FOLD_SEED = 2026
AREA_MODE = "ellipsoid"        # A3
LANDCOVER_SOURCE = "worldcover"  # A4 (fallback "index_rules")
MMU_PX_TILE = 10               # A10 (baseline A only)
WGS84_A = 6378137.0
WGS84_E2 = 0.00669438
EPS = 1e-6

# LightGBM baseline B (§4.2)
LGB_PARAMS = dict(num_leaves=63, learning_rate=0.05, n_estimators=1500, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=100, seed=SEED)
LGB_PIX_PER_TILE = 2000
LGB_CAL_PIX_PER_TILE = 4000

# U-Net (§5.2)
UNET_WIDTHS = (16, 32, 64, 128)
UNET_BOTTLENECK = 256
UNET_DROPOUT = 0.2
UNET_EPOCHS = 40
UNET_BATCH = 16
UNET_LR = 1e-3
UNET_PATIENCE = 8
MODALITY_DROPOUT = 0.25
FOCAL_GAMMA = 2.0
FOCAL_ALPHA1 = 0.75
LAMBDA_A = 1.0
MC_T = 10
CONF_ALPHA = 0.1
SIGMA_FLOOR_KM2 = 0.005
N_GMM = 4
GMM_N_INIT = 20
N_BOOT = 1000
MC_GRADE = 2000

# ---------------- Hengzhou (Q3-Q5) ----------------
A_REG_CITY_KM2 = 3440.4        # GADM boundary area (re-computed geodesically in problem3)
PRE_DATE = "20260627"
FLOOD_DATES = ["20260704", "20260716"]   # A6
MAIN_DATE = "20260716"
S1_SCALE = "auto"              # A5
W0_MODE = "union"              # A7
REM_MAX = 15.0                 # A8 (m)
DELTA_DB = 3.0                 # A9 (dB)
MMU_PX_S1 = 50                 # A10
WATER_DB_MAX = -15.0           # physical bound for accepted S1 VV water thresholds (dB)
Q3_MC_PER_MEMBER = 2           # 5 members x 2 passes = 10
POP_SCALE = 1.0                # A11
FACILITY_PROXY = "town_nodes"  # A12
SPEEDS_KMH = {"motorway": 80, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 30,
              "unclassified": 20, "residential": 20, "service": 20, "living_street": 20, "track": 15}
SPEED_FACTOR = 1.0             # A13
ROAD_BUFFER_M = 20             # A14
ROAD_SAMPLE_M = 20
Q_KG = 3.5                     # A15 kg/person/day
HORIZON_D = 3
CAP_TOWN_T = 60.0              # A16
CAP_CITY_T = 200.0
K_MAX = 6
N_VEH = 40
Q_VEH_T = 5.0
SHIFT_H = 12.0
T_LOAD_MIN = 30.0
T_MAX_MIN = 120.0
ACCESS_MAX_KM = 5.0
ACCESS_KMH = 15.0
DEPOT_ACCESS_MAX_KM = 1.0
DEPOT_FLOOD_CLEAR_M = 100.0
DEPOT_MERGE_M = 300.0
FACILITY_EXPOSE_M = 500.0
AFFECTED_MIN_KM2 = 0.005

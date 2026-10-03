"""项目路径、标签和特征配置。"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
UCI_DIR = RAW_DIR / "UCI HAR Dataset"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

CLASS_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

CLASS_NAMES_ZH = {
    1: "行走",
    2: "上楼",
    3: "下楼",
    4: "坐姿",
    5: "站姿",
    6: "卧姿",
}

SENSOR_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)

METRICS = ("mean", "std", "rms")
FEATURE_NAMES = [
    f"{channel}_{metric}"
    for channel in SENSOR_CHANNELS
    for metric in METRICS
]

FEATURE_UNITS = {
    "body_acc_x": "g",
    "body_acc_y": "g",
    "body_acc_z": "g",
    "body_gyro_x": "rad/s",
    "body_gyro_y": "rad/s",
    "body_gyro_z": "rad/s",
}

METRIC_DESCRIPTIONS = {
    "mean": "窗口内信号的平均水平",
    "std": "窗口内信号的变化程度",
    "rms": "窗口内信号的整体强度",
}

RANDOM_STATE = 42

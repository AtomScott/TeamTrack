from yacs.config import CfgNode as CN

# #############################
# MAIN CONFIG
# #############################
_C = CN()

# #############################
# DATASET CONFIG
# #############################
_C.DATASET = CN()
# The root directory of the dataset
_C.DATASET.ROOT = "/teamtrack/data/teamtrack"
# The name of the dataset
_C.DATASET.NAME = "Basketball_SideView"
# The subset of the dataset to use (train, test, etc.)
_C.DATASET.SUBSET = "train"

# #############################
# OUTPUT CONFIG
# #############################
_C.OUTPUT = CN()
# The root directory where the output will be saved
_C.OUTPUT.ROOT = "/teamtrack/outputs/teamtrack"
# If True, existing output directories will be overwritten
_C.OUTPUT.OVERWRITE = False
# If True, evaluation results will be saved
_C.OUTPUT.SAVE_EVAL = False
# If True, videos will be saved
_C.OUTPUT.SAVE_VIDEO = False

# #############################
# TRACKER CONFIG
# #############################
_C.TRACKER = CN()
# The name of the tracker to be used
_C.TRACKER.NAME = ""

# #############################
# YOLOV8 CONFIG
# #############################
_C.TRACKER.YOLOV8 = CN()
# Tracker type to use with YOLOv8
_C.TRACKER.YOLOV8.TRACKER_TYPE = ""
# High threshold for the first association
_C.TRACKER.YOLOV8.TRACK_HIGH_THRESH = 0.5
# Low threshold for the second association
_C.TRACKER.YOLOV8.TRACK_LOW_THRESH = 0.1
# Threshold for initiating new track if detection doesn't match any tracks
_C.TRACKER.YOLOV8.NEW_TRACK_THRESH = 0.6
# Buffer size to calculate the time when to remove tracks
_C.TRACKER.YOLOV8.TRACK_BUFFER = 30
# Threshold for matching tracks
_C.TRACKER.YOLOV8.MATCH_THRESH = 0.8
# Confidence level for YOLOv8
_C.TRACKER.YOLOV8.CONF = 0.5
# Image size for YOLOv8
_C.TRACKER.YOLOV8.IMGSZ = 640
# Video stride for YOLOv8
_C.TRACKER.YOLOV8.VID_STRIDE = 1
# Global motion compensation method
_C.TRACKER.YOLOV8.CMC_METHOD = "sparseOptFlow"
# Proximity threshold for YOLOv8
_C.TRACKER.YOLOV8.PROXIMITY_THRESH = 0.5
# Appearance threshold for YOLOv8
_C.TRACKER.YOLOV8.APPEARANCE_THRESH = 0.25
# Flag to indicate if re-identification should be used
_C.TRACKER.YOLOV8.WITH_REID = False
# Path to the YOLOv8 model
_C.TRACKER.YOLOV8.MODEL_PATH = "teamtrack/models/yolov5X.pt"
# Configuration path (not used for HPO)
_C.TRACKER.YOLOV8.CONFIGURATION_PATH = './config'

# #############################
# TRACKEVAL CONFIG
# #############################
_C.TRACKEVAL = CN()

# #############################
# EVAL CONFIG
# #############################
_C.TRACKEVAL.EVAL = CN()
# If True, parallel computation will be used
_C.TRACKEVAL.EVAL.USE_PARALLEL = False
# Number of cores to use for parallel computation
_C.TRACKEVAL.EVAL.NUM_PARALLEL_CORES = 8
# If True, the evaluation will stop when an error occurs
_C.TRACKEVAL.EVAL.BREAK_ON_ERROR = True
# If True, the results will be printed
_C.TRACKEVAL.EVAL.PRINT_RESULTS = True
# If True, only the combined results will be printed
_C.TRACKEVAL.EVAL.PRINT_ONLY_COMBINED = False
# If True, the config will be printed
_C.TRACKEVAL.EVAL.PRINT_CONFIG = True
# If True, time progress will be displayed
_C.TRACKEVAL.EVAL.TIME_PROGRESS = True
# If True, a summary of the output will be generated
_C.TRACKEVAL.EVAL.OUTPUT_SUMMARY = True
# If True, detailed output will be generated
_C.TRACKEVAL.EVAL.OUTPUT_DETAILED = True
# If True, curves will be plotted
_C.TRACKEVAL.EVAL.PLOT_CURVES = True

# #############################
# DATASET CONFIG FOR TRACKEVAL
# #############################
_C.TRACKEVAL.DATASET = CN()
# Directory of the ground truth data
_C.TRACKEVAL.DATASET.GT_FOLDER = "./data/teamtrack-mot"
# Directory of the trackers' data
_C.TRACKEVAL.DATASET.TRACKERS_FOLDER = "./outputs/benchmark"
# Directory where the output should be stored
_C.TRACKEVAL.DATASET.OUTPUT_FOLDER = None
# List of trackers to evaluate
_C.TRACKEVAL.DATASET.TRACKERS_TO_EVAL = None
# Classes to evaluate
_C.TRACKEVAL.DATASET.CLASSES_TO_EVAL = ["pedestrian"]
# Benchmark to evaluate
_C.TRACKEVAL.DATASET.BENCHMARK = "Basketball_SideView"
# Split of the dataset to evaluate
_C.TRACKEVAL.DATASET.SPLIT_TO_EVAL = "train"
# If True, input is assumed to be a zip file
_C.TRACKEVAL.DATASET.INPUT_AS_ZIP = False
# If True, the config will be printed
_C.TRACKEVAL.DATASET.PRINT_CONFIG = True
# If True, preprocessing will be done
_C.TRACKEVAL.DATASET.DO_PREPROC = False
# Sub-folder inside the tracker folder
_C.TRACKEVAL.DATASET.TRACKER_SUB_FOLDER = "data"
# Sub-folder inside the output folder
_C.TRACKEVAL.DATASET.OUTPUT_SUB_FOLDER = ""

# #############################
# METRICS CONFIG
# #############################
_C.TRACKEVAL.METRICS = CN()
# Metrics to use for evaluation
_C.TRACKEVAL.METRICS.METRICS = ["HOTA", "CLEAR", "Identity", "VACE"]
# Threshold for the metrics
_C.TRACKEVAL.METRICS.THRESHOLD = 0.5

# #############################
# HYPERPARAMETER OPTIMIZATION CONFIG
# #############################
_C.HPO = CN()
# Number of trials for hyperparameter optimization
_C.HPO.N_TRIALS = 10

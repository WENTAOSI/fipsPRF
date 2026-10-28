
"""One-hemifield FIPS experiment settings.

Scan runs alternate between two types:

Run A: FIPS Left/Right and Frame Only Left/Right.
       20 trials in 5 balanced mini-blocks.
       Duration: 368 s (368 TRs at TR = 1 s).

Run B: Perceptual Left/Right and Veridical.
       15 trials in 5 balanced mini-blocks.
       Duration: 278 s (278 TRs at TR = 1 s).

Each stimulus lasts 10 s, with 8 s ITIs and
8 s pre/post baselines. Scanner TR = 1 s.

Trial counts are configurable, and run durations
are calculated automatically.

Perceptual displacements come from the participant's
prescan summary.
"""


# DEBUG / TASK
CODE_TEST = False
TASK = "FIPS"

# PREVIEW / DIMENSION TESTING
# Preview mode uses test displacements; disable for data collection.
PREVIEW_MODE = False

# Test displacements for preview mode or scans without EyeLink.
DUMMY_SCAN_USE_FAKE_PRESCAN = True
PREVIEW_PRESCAN_LEFT_DVA = 0.75
PREVIEW_PRESCAN_RIGHT_DVA = 0.75


DEFAULT_SUB_ID = "99"
DEFAULT_SUB_NAME = "PT"
DEFAULT_RUN = "2"

# Relative to fips_one_hemi.py.
DATA_DIR = "../data/psycphys"

# BIDS scan events output
BIDS_DATA_DIR = "../data/bids"
BIDS_SESSION_ID = "01"

# MONITOR / WINDOW

MONITORS = {
    "Macbook": {
        "mon_dist": 50,
        "size_cm": (42, 0),
        "size_px": (1440, 900),
        "refresh_rate": 60,
        "screen": 0,
    },
    "iMac": {
        "mon_dist": 60,
        "size_cm": (30, 0),
        "size_px": (1920, 1080),
        "refresh_rate": 60,
        "screen": 0,
    },
    "TseLab": {
        "mon_dist": 63,
        "size_cm": (47.2, 0),
        "size_px": (1920, 1080),
        "refresh_rate": 60,
        "screen": 0,
    },
    "Beaver": {
        "mon_dist": 80,
        "size_cm": (52.3, 0),
        "size_px": (1920, 1080),
        "refresh_rate": 60,
        "screen": 0,
    },
    "Scanner": {
        "mon_dist": 113.7,        # In-bore viewing distance (cm).
        "size_cm": (35.4, 26.5), # Projected image size (cm).
        "size_px": (1024, 768),
        "refresh_rate": 60,
        "screen": 0,
    },
}

DEFAULT_MONITOR = "Scanner"

# Request the pRF display mode; the monitor/GPU supplies side bars.
FULLSCREEN = True
MATCH_PRF_DISPLAY_MODE = True  # Scanner only.
ALLOW_GUI = False

BACK_COLOR = [2 * 127 / 255 - 1] * 3  # pRF gray level: 127/255.

# EYELINK

EYELINK_HOST = "100.1.1.1"
EYELINK_CALIBRATION_TYPE = "HV9"
EYELINK_CALIBRATION_TARGET_SIZE_DVA = 0.25

EYELINK_CALIBRATION_TARGET_SIZE_PX = round(
    EYELINK_CALIBRATION_TARGET_SIZE_DVA
    * MONITORS["Scanner"]["mon_dist"]
    * 0.017455
    * MONITORS["Scanner"]["size_px"][0]
    / MONITORS["Scanner"]["size_cm"][0]
)

DEFAULT_EYELINK_CONNECTED = "yes"

# Optional trial-start fixation gate; recording does not require it.
REQUIRE_FIXATION_BEFORE_TRIAL = False
FIXATION_GATE_DURATION_S = 0.2
FIXATION_GATE_RADIUS_DVA = 1.5
EYE_USED = 1  # 1 = right, 0 = left

# SCANNER

SCANNER_TRIGGER_KEY = "5"
# Acquisition metadata; stimulus durations are in seconds.
SCANNER_TR_S = 1.0

# ONE-HEMIFIELD SPATIAL LAYOUT

# Fixation is 4 dva from the screen edge, mirrored by hemifield.
HEMIFIELD = "right"

# Use the pRF width-based calibration.
FIXATION_ECCENTRICITY_DVA = (
    MONITORS["Scanner"]["size_cm"][0]
    / (2 * MONITORS["Scanner"]["mon_dist"] * 0.017455)
    - 3.0
)

FIXATION_Y_DVA = 0.0
STIMULUS_Y_DVA = 0.0

# Fixation / attention task
FIXATION_DIAMETER_PX = 4
FIXATION_ALPHA = 0.5
FIXATION_COLORS_RGB255 = [
    [255, 0, 0],
    [0, 0, 0],
    [255, 255, 255],
]
FIXATION_COLOR_NAMES = ["red", "black", "white"]

# One random fixation change in approximately 50% of scan blocks.
FIXATION_TASK_HZ = 15.0
FIXATION_BLOCK_CHANGE_FRACTION = 0.5
FIXATION_CHANGE_EDGE_MARGIN_S = 1

FIXATION_RESPONSE_KEYS = ["1", "2", "3", "4"]
FIXATION_RESPONSE_WINDOW_S = 1.0

# Probes
DOT_RADIUS_DVA = 0.7
DOT_VERTICAL_OFFSET_DVA = 3.0

# Moving frame
FRAME_WIDTH_DVA = 7.5
FRAME_HEIGHT_DVA = 9.0
FRAME_LINEWIDTH_DVA = 0.25
FRAME_COLOR = [0.3, 0.3, 0.3]
# Frame bounds include the outline and screen margin.
FRAME_FIXATION_GAP_DVA = 1.0
FRAME_SCREEN_MARGIN_DVA = 0.5
_SCANNER_PPD = (
    MONITORS["Scanner"]["mon_dist"] * 0.017455
    * MONITORS["Scanner"]["size_px"][0]
    / MONITORS["Scanner"]["size_cm"][0]
)
_FRAME_NEAR_EDGE_DVA = (
    -round(FIXATION_ECCENTRICITY_DVA * _SCANNER_PPD) / _SCANNER_PPD
    + FRAME_FIXATION_GAP_DVA
)
_FRAME_FAR_EDGE_DVA = (
    MONITORS["Scanner"]["size_px"][0] / 2 - FRAME_SCREEN_MARGIN_DVA * _SCANNER_PPD
) / _SCANNER_PPD
FRAME_CENTER_DVA = (_FRAME_NEAR_EDGE_DVA + _FRAME_FAR_EDGE_DVA) / 2
STIMULUS_ECCENTRICITY_DVA = FRAME_CENTER_DVA
FRAME_TRAVEL_DVA = (
    _FRAME_FAR_EDGE_DVA - _FRAME_NEAR_EDGE_DVA
    - FRAME_WIDTH_DVA - FRAME_LINEWIDTH_DVA - 1.0 / _SCANNER_PPD
)
if FRAME_TRAVEL_DVA <= 0:
    raise ValueError("Frame cannot fit between fixation clearance and screen edge")

STIMULUS_CHOICES = ["checkerboard", "plain", "old"]
DEFAULT_STIMULUS = "plain"

# TEMPORAL PARAMETERS

PRESCAN = {
    "n_trials": 16,
    "cycle": 8,
    "pre_run_baseline": 0.0,
    "post_run_baseline": 0.0,
    "dot_duration": 50 / 1000,
    "reversal_time": 450 / 1000,
    "ITI": 2 / 1000,
}


SCAN = {
    "cycle": 10,
    "pre_run_baseline": 8.0,
    "post_run_baseline": 8.0,
    "dot_duration": 50 / 1000,
    "reversal_time": 450 / 1000,
    "ITI": 8.0,
}


# GLOBAL SCANNER RUN SCHEDULE
# pRF runs are executed separately on another computer.
SCAN_RUN_TYPES = {
    1: "pRF",
    2: "A",
    3: "B",
    4: "A",
    5: "pRF",
    6: "B",
    7: "A",
    8: "B",
    9: "pRF",
    10: "A",
    11: "B",
    12: "A",
    13: "pRF",
    14: "B",
}

FIPS_RUN_NUMBERS = [
    number
    for number, run_type in SCAN_RUN_TYPES.items()
    if run_type in ("A", "B")
]

# Enable or disable inter-trial intervals independently.
RUN_A_ITI_ENABLED = True
RUN_B_ITI_ENABLED = True


# 450 ms motion + 50 ms flash per half-cycle.
# Scan: 10 cycles (10 s); prescan: 8 cycles (8 s).

# SCAN CONDITIONS

COND_FIPS = "FIPS"
COND_VERIDICAL = "Veridical"
COND_PERCEPTUAL = "Perceptual"
COND_FRAME_ONLY = "FrameOnly"

# NUMBER OF TRIALS PER CONDITION PER RUN
RUN_A_TRIAL_COUNTS = {
    "fips_left": 5,
    "fips_right": 5,
    "frame_only_starts_left": 5,
    "frame_only_starts_right": 5,
}

RUN_B_TRIAL_COUNTS = {
    "perceptual_left": 5,
    "perceptual_right": 5,
    "veridical": 5,
}

# TESTING
CODE_TEST_N_TRIALS = 4
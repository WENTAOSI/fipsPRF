#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-hemifield FIPS experiment with scanner triggering and EyeLink recording."""

# DEBUG / IMPORTS

from psychopy import visual, event, core, monitors, logging, gui, data
from psychopy.tools.attributetools import undefined
import numpy as np
import copy
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

import fips_config as cfg
import atexit
import pyglet
from pyglet import gl


def select_display_mode(screen_index, size_px, refresh_hz):
    """Request the pRF display resolution and refresh rate."""
    screens = pyglet.canvas.get_display().get_screens()
    if not 0 <= screen_index < len(screens):
        raise ValueError(f"Display index {screen_index} is unavailable")
    screen = screens[screen_index]
    width, height = map(int, size_px)
    matches = [mode for mode in screen.get_modes()
               if (mode.width, mode.height, mode.rate) ==
               (width, height, refresh_hz)]
    if not matches:
        logging.warning("Requested pRF display mode unavailable; using scaled 4:3 fallback")
        return None
    mode = max(matches, key=lambda item: item.depth)
    screen.set_mode(mode)
    actual = screen.get_mode()
    if (actual.width, actual.height, actual.rate) != (width, height, refresh_hz):
        screen.restore_mode()
        logging.warning("Windows refused pRF display mode; using scaled 4:3 fallback")
        return None
    return screen


class PRFMatchedWindow(visual.Window):
    """Restore the display mode when the fullscreen window closes."""

    def __init__(self, *args, match_display_mode=False, refresh_hz=60,
                 preview_canvas_px=None, **kwargs):
        self._closed = True
        self._matched_screen = None
        self.letterbox_rect = None
        self.render_scale = 1.0
        if match_display_mode:
            self._matched_screen = select_display_mode(
                kwargs.get("screen", 0), kwargs["size"], refresh_hz
            )
            atexit.register(self.restore_display_mode)
        try:
            super().__init__(*args, **kwargs)
            if match_display_mode:
                actual_w, actual_h = map(int, self.frameBufferSize)
                target_w, target_h = kwargs["size"]
                self.render_scale = min(actual_w / target_w, actual_h / target_h)
                if (actual_w, actual_h) != (target_w, target_h):
                    canvas_w = round(target_w * self.render_scale)
                    canvas_h = round(target_h * self.render_scale)
                    self.letterbox_rect = ((actual_w-canvas_w)//2,
                                          (actual_h-canvas_h)//2,
                                          canvas_w, canvas_h)
                    self._endOfFlip(True)
            elif preview_canvas_px is not None:
                # Scissor coordinates use framebuffer pixels; Retina drawing units are halved.
                actual_w, actual_h = map(int, self.frameBufferSize)
                target_w, target_h = preview_canvas_px
                framebuffer_scale = min(actual_w / target_w, actual_h / target_h)
                self.render_scale = framebuffer_scale / (2 if self.useRetina else 1)
                canvas_w = round(target_w * framebuffer_scale)
                canvas_h = round(target_h * framebuffer_scale)
                self.letterbox_rect = ((actual_w - canvas_w) // 2,
                                      (actual_h - canvas_h) // 2,
                                      canvas_w, canvas_h)
                self._endOfFlip(True)
        except BaseException:
            self.restore_display_mode()
            raise

    def _endOfFlip(self, clearBuffer):
        if self.letterbox_rect is not None:
            if clearBuffer:
                self.scissorTest = False
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                self.color = self.color
            self.scissor = self.letterbox_rect
            self.scissorTest = True
        super()._endOfFlip(clearBuffer)

    def restore_display_mode(self):
        if self._matched_screen is not None:
            screen, self._matched_screen = self._matched_screen, None
            screen.restore_mode()

    def close(self):
        try:
            super().close()
        finally:
            self.restore_display_mode()


CODE_TEST = cfg.CODE_TEST
DEBUG_ON_VSCODE = cfg.DEBUG_ON_VSCODE
TASK = cfg.TASK


# SESSION INFO / DATA PATHS

expInfo = {
    "sub_id": cfg.DEFAULT_SUB_ID,
    "sub_name": cfg.DEFAULT_SUB_NAME,
    "run": cfg.DEFAULT_RUN,
}

ROOTDIR = Path(__file__).resolve().parent
DATADIR = (ROOTDIR / cfg.DATA_DIR).resolve()
DATADIR.mkdir(parents=True, exist_ok=True)

# Automatically select the next scan run from existing BIDS files.
bids_root = (ROOTDIR / cfg.BIDS_DATA_DIR).resolve()
bids_session = str(cfg.BIDS_SESSION_ID).zfill(2)

existing_events = list(bids_root.glob(
    f"sub-*/ses-{bids_session}/"
    f"sub-*_ses-{bids_session}_task-{TASK}_run-*_events.tsv"
))

if existing_events:
    latest_file = max(
        existing_events,
        key=lambda path: path.stat().st_mtime
    )

    latest_subject = latest_file.parent.parent.name
    expInfo["sub_id"] = latest_subject.removeprefix("sub-")

    subject_dir = bids_root / latest_subject / f"ses-{bids_session}"

    existing_runs = []

    for path in subject_dir.glob(
        f"{latest_subject}_ses-{bids_session}_task-{TASK}_run-*_events.tsv"
    ):
        run_text = path.name.rsplit("_run-", 1)[-1].removesuffix(
            "_events.tsv"
        )

        if run_text.isdigit():
            existing_runs.append(int(run_text))

    if existing_runs:
        expInfo["run"] = str(max(existing_runs) + 1)

# GUI

dlg = gui.Dlg(title=TASK)
dlg.addField("Subject ID:", expInfo["sub_id"])
dlg.addField("Initials:", expInfo["sub_name"])
dlg.addField("Session:", choices=["prescan", "scan"], initial="scan")
dlg.addField(
    "Monitor:",
    choices=list(cfg.MONITORS.keys()),
    initial=cfg.DEFAULT_MONITOR,
)
dlg.addField(
    "Stimulus:",
    choices=cfg.STIMULUS_CHOICES,
    initial=cfg.DEFAULT_STIMULUS,
)
dlg.addField(
    "Eyelink Connected:",
    choices=["no", "yes"],
    initial=cfg.DEFAULT_EYELINK_CONNECTED,
)
dlg.addField("Run:", expInfo["run"])
dlg.show()

if dlg.OK is False:
    core.quit()

expInfo.update(
    {
        "exp_name": TASK,
        "sub_id": dlg.data[0],
        "sub_name": dlg.data[1],
        "ses": dlg.data[2],
        "monitor": dlg.data[3],
        "stimulus": dlg.data[4],
        "eyelink": dlg.data[5],
        "run": dlg.data[6],
        "date": data.getDateStr(),
    }
)

# Confirm unusual scan run selections.
if expInfo["ses"] == "scan":
    selected_run = int(expInfo["run"])
    selected_sub = f"{int(expInfo['sub_id']):02d}"

    warnings = []

    # Warn if the run exceeds the configured maximum.
    if selected_run > cfg.SCAN["n_runs"]:
        warnings.append(
            f"Run {selected_run} exceeds the configured maximum "
            f"of {cfg.SCAN['n_runs']} runs."
        )

    # Warn if the immediately preceding run does not exist.
    if selected_run > 1:
        previous_run = selected_run - 1
        previous_run_id = f"{previous_run:02d}"

        previous_events = (
            bids_root
            / f"sub-{selected_sub}"
            / f"ses-{bids_session}"
            / f"sub-{selected_sub}_ses-{bids_session}_task-{TASK}_run-{previous_run_id}_events.tsv"
        )

        if not previous_events.is_file():
            warnings.append(
                f"Run {previous_run} does not exist "
                f"for subject {selected_sub}, session {bids_session}."
            )

    # Show one confirmation popup if any warning applies.
    if warnings:
        confirmation = gui.Dlg(title="Confirm Scan Run")

        confirmation.addText(
            "\n\n".join(warnings)
            + "\n\nDo you want to continue with this run?"
        )

        confirmation.show()

        if not confirmation.OK:
            core.quit()


FLASH_GRAB = expInfo["stimulus"] == "checkerboard"
EYELINK_CONNECTED = expInfo["eyelink"] == "yes"

# Dummy scans can use test displacements without a prescan.
DUMMY_SCAN = (
    expInfo["ses"] == "scan"
    and not EYELINK_CONNECTED
)


# SAVING / LOGGING

sub_id = f"{int(expInfo['sub_id']):02d}"
run_id = f"{int(expInfo['run']):02d}"

SUBDIR = DATADIR / f"sub-{sub_id}"
EDFDIR = DATADIR / "edf"

for directory in [SUBDIR, EDFDIR]:
    directory.mkdir(parents=True, exist_ok=True)

BIDS_EVENTS_NAME = None

if expInfo["ses"] == "scan":
    bids_session = str(cfg.BIDS_SESSION_ID).zfill(2)

    bids_dir = (
        ROOTDIR / cfg.BIDS_DATA_DIR
        / f"sub-{sub_id}" / f"ses-{bids_session}"
    ).resolve()

    bids_dir.mkdir(parents=True, exist_ok=True)

    BIDS_EVENTS_NAME = (
        bids_dir
        / f"sub-{sub_id}_ses-{bids_session}_task-{TASK}_run-{run_id}_events.tsv"
    )

    if BIDS_EVENTS_NAME.exists():
        raise FileExistsError(
            f"BIDS events file already exists: {BIDS_EVENTS_NAME}\n"
            "Choose a different run number to avoid overwriting data."
        )

class _NoFileLog:
    def write(self, message):
        pass

logFile = _NoFileLog()
logging.console.setLevel(logging.WARNING)

# EyeLink Host filenames must not exceed 8 characters.
edf_task = "PRES" if expInfo["ses"] == "prescan" else TASK[:4].upper()
edf_fname = f"{sub_id}{edf_task}{run_id}"

if len(edf_fname) > 8:
    raise ValueError(f"EyeLink filename exceeds 8 characters: {edf_fname}")

edf_file = edf_fname + ".EDF"

# Include the run to avoid overwriting local EDF files.
session_identifier = (
    f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_{expInfo['ses']}_run-{run_id}"
)
session_folder = EDFDIR / session_identifier
session_folder.mkdir(parents=True, exist_ok=True)
local_edf = session_folder / f"{session_identifier}.EDF"

# LOAD PRESCAN SUMMARY
PRESCAN_LEFT = 0.0
PRESCAN_RIGHT = 0.0
PRESCAN_AVG = 0.0

predataName = (
    SUBDIR
    / f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_prescan_summary.csv"
)

# Prevent accidental overwriting of existing prescan data.
if expInfo["ses"] == "prescan" and predataName.exists():
    raise FileExistsError(
        f"Prescan data already exists:\n{predataName}\n\n"
        "Back up or rename the existing file before "
        "starting a new prescan."
    )

use_fake_prescan = (
    cfg.PREVIEW_MODE
    or (
        DUMMY_SCAN
        and cfg.DUMMY_SCAN_USE_FAKE_PRESCAN
    )
)

if use_fake_prescan:
    PRESCAN_LEFT = float(cfg.PREVIEW_PRESCAN_LEFT_DVA)
    PRESCAN_RIGHT = float(cfg.PREVIEW_PRESCAN_RIGHT_DVA)
    PRESCAN_AVG = (PRESCAN_LEFT + PRESCAN_RIGHT) / 2.0

    fake_mode_name = (
        "DUMMY SCAN"
        if DUMMY_SCAN and not cfg.PREVIEW_MODE
        else "PREVIEW MODE"
    )

    prescan_text = (
        f"{fake_mode_name}: using fake prescan displacement.\n"
        f"Left: {PRESCAN_LEFT:.3f} dva\n"
        f"Right: {PRESCAN_RIGHT:.3f} dva\n"
        f"Average: {PRESCAN_AVG:.3f} dva\n"
        "TESTING ONLY - NOT FOR DATA COLLECTION."
    )

    print(f"\n*** {fake_mode_name} - FAKE PRESCAN DVA ***")
    print(prescan_text)
    print("**************************************\n")


elif (
    expInfo["ses"] == "scan"
    and cfg.SCAN_RUN_TYPES[
        (int(expInfo["run"]) - 1) % len(cfg.SCAN_RUN_TYPES)
    ] == "A"
):
    prescan_text = "Run A: prescan measurements not required."

elif predataName.exists():
    prescan_summary_df = pd.read_csv(predataName)
    if (
        expInfo["ses"] == "scan"
        and cfg.SCAN_RUN_TYPES[
            (int(expInfo["run"]) - 1) % len(cfg.SCAN_RUN_TYPES)
        ] == "B"
    ):

        required = ["left", "right", "avg"]

        if (
            len(prescan_summary_df) != cfg.PRESCAN["n_trials"]
            or not all(c in prescan_summary_df.columns for c in required)
            or prescan_summary_df.loc[0, required].isna().any()
        ):
            raise RuntimeError(
                f"Prescan data is incomplete: {predataName}"
            )
    PRESCAN_LEFT = float(prescan_summary_df.loc[0, "left"])
    PRESCAN_RIGHT = float(prescan_summary_df.loc[0, "right"])
    PRESCAN_AVG = float(prescan_summary_df.loc[0, "avg"])

    # Validate participant-specific displacements before Run B.
    if (
        expInfo["ses"] == "scan"
        and cfg.SCAN_RUN_TYPES[
            (int(expInfo["run"]) - 1) % len(cfg.SCAN_RUN_TYPES)
        ] == "B"
    ):
        values = [PRESCAN_LEFT, PRESCAN_RIGHT, PRESCAN_AVG]

        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "Invalid prescan displacement values. "
                "All values must be finite and nonnegative."
            )

        if PRESCAN_LEFT == 0 and PRESCAN_RIGHT == 0:
            raise ValueError(
                "Both prescan displacements are zero. "
                "Verify the participant's prescan measurements."
            )

    prescan_text = (
        "Loaded prescan summary:\n"
        f"Left: {PRESCAN_LEFT:.3f} dva\n"
        f"Right: {PRESCAN_RIGHT:.3f} dva\n"
        f"Average: {PRESCAN_AVG:.3f} dva"
    )

else:
    prescan_text = f"Prescan summary file not found:\n{predataName}"


# MONITOR / WINDOW

monName = expInfo["monitor"]
monInfo = cfg.MONITORS[monName]

expMon = monitors.Monitor(
    name=monName,
    width=monInfo["size_cm"][0],
    distance=monInfo["mon_dist"],
)
expMon.setSizePix(monInfo["size_px"])

expWin = PRFMatchedWindow(
    match_display_mode=(cfg.FULLSCREEN and cfg.MATCH_PRF_DISPLAY_MODE and monName == "Scanner"),
    preview_canvas_px=(cfg.MONITORS["Scanner"]["size_px"] if monName == "Macbook" else None),
    refresh_hz=monInfo["refresh_rate"],
    monitor=expMon,
    size=monInfo["size_px"],
    screen=monInfo["screen"],
    fullscr=cfg.FULLSCREEN,
    units="pix",
    winType="pyglet",
    allowGUI=cfg.ALLOW_GUI,
    allowStencil=False,
    color=[-1, -1, -1],
    colorSpace="rgb",
    blendMode="avg",
    waitBlanking=True,
)

logFile.write(f"Actual display pixels = {tuple(expWin.frameBufferSize)}\n")

logFile.write(f"Monitor: {monName}\n")
logFile.write(
    f"Distance = {monInfo['mon_dist']} cm, "
    f"Width = {monInfo['size_cm'][0]} cm\n"
)
logFile.write(
    f"Pixel Width = {monInfo['size_px'][0]}, "
    f"Pixel Height = {monInfo['size_px'][1]}\n"
)
logFile.write(f"Refresh Rate = {monInfo['refresh_rate']} Hz\n")

screen_width_cm = monInfo["size_cm"][0]
screen_width_px = monInfo["size_px"][0]
viewing_distance_cm = monInfo["mon_dist"]

px_per_cm = screen_width_px / screen_width_cm
# Use the pRF pixel-to-degree conversion.
cm_per_dva = viewing_distance_cm * 0.017455
# Scale the logical canvas if the requested display mode is unavailable.
render_scale = expWin.render_scale
ppd = cm_per_dva * px_per_cm * render_scale
if monName == "Macbook":
    # Keep laptop preview geometry consistent with Scanner.
    scanner_info = cfg.MONITORS["Scanner"]
    ppd = (
        scanner_info["mon_dist"] * 0.017455
        * scanner_info["size_px"][0] / scanner_info["size_cm"][0]
        * render_scale
    )
logFile.write(f"pRF render scale = {render_scale}; canvas = {expWin.letterbox_rect}\n")


def deg2pix(dva):
    if isinstance(dva, (list, tuple, np.ndarray)):
        return [x * ppd for x in dva]
    return dva * ppd


# EYELINK SETUP

if EYELINK_CONNECTED:
    try:
        el_tracker = pylink.EyeLink(cfg.EYELINK_HOST)
    except RuntimeError as error:
        print("ERROR:", error)
        expWin.close()
        core.quit()
        sys.exit()
else:
    el_tracker = pylink.EyeLink(None)

try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print("ERROR:", err)
    if el_tracker.isConnected():
        el_tracker.close()
    expWin.close()
    core.quit()
    sys.exit()

experiment_info = (
    f"RECORDED BY {os.path.basename(__file__)}\n"
    f"Subject ID: {expInfo['sub_id']}\n"
    f"Session: {expInfo['ses']}\n"
    f"Run: {expInfo['run']}\n"
    f"Hemifield: {cfg.HEMIFIELD}\n"
    f"Monitor: {expInfo['monitor']}\n"
    f"Date: {expInfo['date']}"
)
el_tracker.sendCommand(f"add_file_preamble_text '{experiment_info}'")

el_tracker.setOfflineMode()

eyelink_ver = 0
if EYELINK_CONNECTED:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split(".")[0])
    print("Running experiment on %s, version %d" % (vstr, eyelink_ver))

file_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT"
link_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT"

if eyelink_ver > 3:
    file_sample_flags = (
        "LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT"
    )
    link_sample_flags = (
        "LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT"
    )
else:
    file_sample_flags = (
        "LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT"
    )
    link_sample_flags = (
        "LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT"
    )

el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
el_tracker.sendCommand(f"file_sample_data = {file_sample_flags}")
el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
el_tracker.sendCommand(f"link_sample_data = {link_sample_flags}")

el_tracker.sendCommand(
    f"calibration_type = {cfg.EYELINK_CALIBRATION_TYPE}"
)
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

scn_width, scn_height = expWin.size

el_coords = (
    "screen_pixel_coords = 0 0 %d %d"
    % (scn_width - 1, scn_height - 1)
)
el_tracker.sendCommand(el_coords)

dv_coords = (
    "DISPLAY_COORDS  0 0 %d %d"
    % (scn_width - 1, scn_height - 1)
)
el_tracker.sendMessage(dv_coords)

genv = EyeLinkCoreGraphicsPsychoPy(
    el_tracker,
    expWin,
    disableAudio=not EYELINK_CONNECTED,
)
print(genv)

foreground_color = (-1, -1, -1)
background_color = tuple(cfg.BACK_COLOR)

genv.setCalibrationColors(foreground_color, background_color)
genv.setTargetType("circle")
genv.setTargetSize(cfg.EYELINK_CALIBRATION_TARGET_SIZE_PX)
genv.setCalibrationSounds("", "", "")
pylink.openGraphicsEx(genv)


# EXPERIMENT PARAMETERS

exp = copy.deepcopy(
    cfg.SCAN if expInfo["ses"] == "scan" else cfg.PRESCAN
)

if CODE_TEST:
    exp["n_trials"] = cfg.CODE_TEST_N_TRIALS


# TRIAL GENERATION

def source_initial_direction(induced_percept):
    """Frame direction for upper-first flashes and the requested tilt."""
    return "right" if induced_percept == "left" else "left"

def build_scan_trials(run_number):
    """Build balanced, randomized mini-blocks for Run A or B."""

    run_type = cfg.SCAN_RUN_TYPES[
        (run_number - 1) % len(cfg.SCAN_RUN_TYPES)
    ]

    if run_type == "A":
        conditions = cfg.RUN_A_CONDITIONS
        counts = cfg.RUN_A_TRIAL_COUNTS
    elif run_type == "B":
        conditions = cfg.RUN_B_CONDITIONS
        counts = cfg.RUN_B_TRIAL_COUNTS
    else:
        raise ValueError(f"Unknown run type: {run_type}")

    if set(conditions) != set(counts):
        raise ValueError(
            f"Run {run_type}: conditions and counts do not match"
        )

    if any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("Trial counts must be nonnegative integers")

    n_blocks = max(counts.values(), default=0)

    if n_blocks == 0:
        raise ValueError(f"Run {run_type} has no trials")

    def make_trial(label):
        if label == "veridical":
            return {
                "ExpCondition": cfg.COND_VERIDICAL,
                "InitialPosition": "upper",
                "InducedPercept": "none",
                "InitialDirection": "NA",
            }

        kind, _, side = label.rpartition("_")

        if side not in ("left", "right"):
            raise ValueError(f"Unknown condition: {label}")

        if kind == "fips":
            direction = source_initial_direction(side)
            frame_start = "left" if direction == "right" else "right"

            return {
                "ExpCondition": cfg.COND_FIPS,
                "InitialPosition": "upper",
                "InducedPercept": side,
                "InitialDirection": direction,
                "FrameStartPosition": frame_start,
                "FIPSCategory": (
                    f"{side}_tilt_frame_starts_{frame_start}"
                ),
            }

        if kind == "frame_only_starts":
            direction = "right" if side == "left" else "left"

            return {
                "ExpCondition": cfg.COND_FRAME_ONLY,
                "InitialPosition": "NA",
                "InducedPercept": "none",
                "InitialDirection": direction,
                "FrameStartPosition": side,
            }

        if kind == "perceptual":
            return {
                "ExpCondition": cfg.COND_PERCEPTUAL,
                "InitialPosition": "upper",
                "InducedPercept": side,
                "InitialDirection": "NA",
            }

        raise ValueError(f"Unknown condition: {label}")

    # Distribute trials across balanced mini-blocks.
    blocks = [[] for _ in range(n_blocks)]

    for label in conditions:
        count = counts[label]

        if count == 0:
            continue

        candidates = np.random.permutation(n_blocks).tolist()
        candidates.sort(key=lambda index: len(blocks[index]))

        for block_index in candidates[:count]:
            blocks[block_index].append(make_trial(label))

    # Randomize trials within each mini-block.
    trials = []

    for block_number, block_trials in enumerate(blocks, start=1):
        np.random.shuffle(block_trials)

        for trial in block_trials:
            trial["Block"] = block_number
            trial["Run"] = run_number
            trial["RunType"] = run_type
            trial["Trial"] = len(trials) + 1
            trials.append(trial)

    return trials

def build_prescan_trials():
    factor_prescan = {
        "ExpCondition": [cfg.COND_FIPS],
        "InitialPosition": ["upper"],
        "InducedPercept": ["left", "right"],
    }

    cond_list = data.createFactorialTrialList(factor_prescan)
    n_reps = max(round(exp["n_trials"] // len(cond_list)), 1)
    trials = [
        copy.deepcopy(trial)
        for trial in cond_list
        for _ in range(n_reps)
    ]
    trials = np.random.permutation(trials).tolist()
    trials = trials[: exp["n_trials"]]

    for idx, trial in enumerate(trials, start=1):
        trial["Block"] = 1
        trial["Run"] = int(expInfo["run"])
        trial["Trial"] = idx
        trial["InitialDirection"] = source_initial_direction(
            trial["InducedPercept"],
        )

    return [copy.deepcopy(t) for t in trials]


if expInfo["ses"] == "scan":
    TRIALS = build_scan_trials(int(expInfo["run"]))
else:
    TRIALS = build_prescan_trials()

if CODE_TEST:
    TRIALS = TRIALS[: cfg.CODE_TEST_N_TRIALS]
    for idx, trial in enumerate(TRIALS, start=1):
        trial["Trial"] = idx


# ONE-HEMIFIELD STIMULUS CONFIGURATION

side_sign = 1 if cfg.HEMIFIELD == "right" else -1

FIX_POS_DVA = [
    -side_sign * cfg.FIXATION_ECCENTRICITY_DVA,
    cfg.FIXATION_Y_DVA,
]
STIM_POS_DVA = [
    side_sign * cfg.STIMULUS_ECCENTRICITY_DVA,
    cfg.STIMULUS_Y_DVA,
]

stim_config = {
    "backColor": cfg.BACK_COLOR,
    "session": expInfo["ses"],
    # pRF fixation offsets are rounded to integer pixels.
    "fix_pos": (np.round(np.array(deg2pix(FIX_POS_DVA)) / render_scale) * render_scale).tolist(),
    "fix_diameter_px": cfg.FIXATION_DIAMETER_PX * render_scale,
    "fix_alpha": cfg.FIXATION_ALPHA,
    "stim_pos": deg2pix(STIM_POS_DVA),
    "mark_pos": deg2pix(STIM_POS_DVA),
    "dot_radius": deg2pix(cfg.DOT_RADIUS_DVA),
    "dot_diameter": deg2pix(cfg.DOT_DIAMETER_DVA),
    "center_offset_y": deg2pix(cfg.DOT_VERTICAL_OFFSET_DVA),
    "frame_width": deg2pix(cfg.FRAME_WIDTH_DVA),
    "frame_height": deg2pix(cfg.FRAME_HEIGHT_DVA),
    "frame_linewidth": deg2pix(cfg.FRAME_LINEWIDTH_DVA),
    "frame_color": cfg.FRAME_COLOR,
    "frame_travel": deg2pix(cfg.FRAME_TRAVEL_DVA),
    "frame_center_x": deg2pix(side_sign * cfg.FRAME_CENTER_DVA),
    "frame_per_sec": monInfo["refresh_rate"],
    "sec_per_frame": 1 / monInfo["refresh_rate"],
    "n_frames_per_move": (
        exp["reversal_time"] * monInfo["refresh_rate"]
    ),
    "n_frames_per_pause": (
        exp["dot_duration"] * monInfo["refresh_rate"]
    ),
}


stim_config["shift_pix_per_frame"] = (
    stim_config["frame_travel"]
    / stim_config["n_frames_per_move"]
)
stim_config["shift_pix_per_second"] = (
    stim_config["shift_pix_per_frame"]
    * stim_config["frame_per_sec"]
)

stim_config["duration_one_way"] = (
    exp["reversal_time"] + exp["dot_duration"]
)
stim_config["duration_stimulus"] = (
    stim_config["duration_one_way"] * 2 * exp["cycle"]
)

if expInfo["ses"] == "scan":
    planned_run_s = (
        exp["pre_run_baseline"]
        + len(TRIALS) * stim_config["duration_stimulus"]
        + max(len(TRIALS) - 1, 0) * exp["ITI"]
        + exp["post_run_baseline"]
    )
    logFile.write(
        f"Scanner TR = {cfg.SCANNER_TR_S:.3f} s; "
        f"planned run = {planned_run_s:.3f} s "
        f"({planned_run_s / cfg.SCANNER_TR_S:g} TRs). "
        "Stimulus schedule uses seconds from the first trigger.\n"
    )

expWin.colorSpace = "rgb"
expWin.color = stim_config["backColor"]
expWin.refreshThreshold = (
    1 / monInfo["refresh_rate"] + 0.003
)
expWin.units = "pix"


# GENERATE FIXATION

# pRF fixation is a 4-pixel disc without a border.
fix_outer = visual.Circle(
    win=expWin,
    radius=stim_config["fix_diameter_px"] / 2.0,
    pos=stim_config["fix_pos"],
    fillColor=cfg.FIXATION_COLORS_RGB255[0],
    lineColor=cfg.FIXATION_COLORS_RGB255[0],
    colorSpace="rgb255",
    opacity=stim_config["fix_alpha"],
    autoLog=False,
)

# Invisible inner fixation placeholder.
fix_inner = visual.Circle(
    win=expWin,
    radius=0,
    pos=stim_config["fix_pos"],
    fillColor=[0, 0, 0],
    lineColor=[0, 0, 0],
    colorSpace="rgb255",
    opacity=0,
    autoLog=False,
)


# GENERATE DOTS / MARKERS

dot_positions = [
    [
        stim_config["stim_pos"][0],
        stim_config["stim_pos"][1] + stim_config["center_offset_y"],
    ],
    [
        stim_config["stim_pos"][0],
        stim_config["stim_pos"][1] - stim_config["center_offset_y"],
    ],
]

marker_positions = [
    [
        stim_config["mark_pos"][0],
        stim_config["mark_pos"][1] + stim_config["center_offset_y"],
    ],
    [
        stim_config["mark_pos"][0],
        stim_config["mark_pos"][1] - stim_config["center_offset_y"],
    ],
]

dots = [
    visual.Circle(
        win=expWin,
        radius=stim_config["dot_radius"],
        pos=pos,
        fillColor="white",
        lineColor="white",
        autoLog=False,
    )
    for pos in dot_positions
]

markers = [
    visual.Circle(
        win=expWin,
        radius=stim_config["dot_radius"],
        pos=pos,
        fillColor="white",
        lineColor="white",
        autoLog=False,
    )
    for pos in marker_positions
]

mouseResp = event.Mouse()
mouseResp.autoLog = False


def make_checker_dot(win, pos, radius):
    colors = [
        ["black", 0, 90],
        ["white", 90, 180],
        ["black", 180, 270],
        ["white", 270, 360],
    ]

    pies = []
    for col, ang1, ang2 in colors:
        pie = visual.Pie(
            win,
            radius=radius,
            start=ang1,
            end=ang2,
            fillColor=col,
            lineColor=col,
            pos=pos,
            autoLog=False,

            # The undefined sentinel prevents legacy Pie colors overriding colorSpace.
            colorSpace="rgb",
            color=undefined,
            fillColorSpace=undefined,
            lineColorSpace=undefined,
            lineRGB=undefined,
            fillRGB=undefined,
        )
        pie.base_start = ang1
        pie.base_end = ang2
        pies.append(pie)

    return pies


if FLASH_GRAB:
    dots = [
        make_checker_dot(expWin, pos, stim_config["dot_radius"])
        for pos in dot_positions
    ]
    markers = [
        make_checker_dot(expWin, pos, stim_config["dot_radius"])
        for pos in marker_positions
    ]


# FRAME

frame_outer = visual.Rect(
    win=expWin,
    width=stim_config["frame_width"] + stim_config["frame_linewidth"],
    height=stim_config["frame_height"] + stim_config["frame_linewidth"],
    lineColor=stim_config["frame_color"],
    lineWidth=render_scale if monName == "Macbook" else 1,
    interpolate=False,
    fillColor=stim_config["frame_color"],
    autoLog=False,
    depth=2,
)

frame_inner = visual.Rect(
    win=expWin,
    width=stim_config["frame_width"],
    height=stim_config["frame_height"],
    lineColor=stim_config["backColor"],
    lineWidth=render_scale if monName == "Macbook" else 1,
    interpolate=False,
    fillColor=stim_config["backColor"],
    autoLog=False,
    depth=1,
)


# INSTRUCTIONS

task_text = (
    "Please keep your eyes on the fixation dot.\n\n"
    "Press a response button whenever the fixation dot changes color.\n\n"
    "Press any key to continue."
)

end_run_text = (
    "You've completed this run!\n\n"
    "Take a short break. The next run will start soon."
)
end_end_text = (
    "All done!\n\n"
    "Please stay still and wait for the experimenter."
)


# STIMULUS HELPERS

def draw_dot(dot, draw_or_not=True):
    if FLASH_GRAB:
        for pie in dot:
            pie.autoDraw = draw_or_not
    else:
        dot.autoDraw = draw_or_not


def color_dot(dot, color):
    if FLASH_GRAB:
        opposite = "black" if color == "white" else "white"

        if color == "red":
            opposite = "white"
        elif color == "white_red":
            color = "white"
            opposite = "red"

        for i, pie in enumerate(dot):
            c = color if i % 2 == 0 else opposite
            pie.fillColor = c
            pie.lineColor = c
    else:
        solid_color = "white" if expInfo["stimulus"] == "plain" else color
        dot.fillColor = solid_color
        dot.lineColor = solid_color


def update_dots(pair="upper", color="white", sets=None):
    if sets is None:
        sets = dots

    if pair == "upper":
        color_dot(sets[0], color)
        draw_dot(sets[0], True)
        draw_dot(sets[1], False)

    elif pair == "lower":
        color_dot(sets[1], color)
        draw_dot(sets[1], True)
        draw_dot(sets[0], False)


def hide_all_dots(sets=None):
    if sets is None:
        sets = dots
    for dot in sets:
        draw_dot(dot, False)


def update_dots_by_clock(
    time_elapsed,
    dot_color,
    dot_counter=0,
):
    n_one_way, time_left = divmod(
        time_elapsed,
        stim_config["duration_one_way"],
    )

    is_lower = int(n_one_way) % 2 == 1
    current_pos = "lower" if is_lower else "upper"
    current_color = dot_color

    if time_left > exp["reversal_time"]:
        if is_lower:
            current_color = (
                "black" if dot_color == "white" else "white"
            )
        if FLASH_GRAB:
            dot_counter += 1
        update_dots(current_pos, current_color)
    else:
        hide_all_dots()

    return current_color, dot_counter


def update_markers_by_clock(
    time_elapsed,
    dot_color,
    dot_pos,
):
    n_one_way, time_left = divmod(
        time_elapsed,
        stim_config["duration_one_way"],
    )

    current_color = dot_color
    current_pos = dot_pos

    if time_left > exp["reversal_time"]:
        if int(n_one_way) % 2 == 1:
            current_pos = (
                "upper" if dot_pos == "lower" else "lower"
            )

        if FLASH_GRAB:
            current_color = (
                "black" if dot_color == "white" else "white"
            )

        update_dots(
            current_pos,
            current_color,
            markers,
        )

    else:
        hide_all_dots(markers)

    return current_color


def move_dots(distance):
    """Apply opposite horizontal offsets to the upper and lower probes."""
    pos_list = [
        [
            stim_config["stim_pos"][0] - distance,
            stim_config["stim_pos"][1] + stim_config["center_offset_y"],
        ],
        [
            stim_config["stim_pos"][0] + distance,
            stim_config["stim_pos"][1] - stim_config["center_offset_y"],
        ],
    ]

    for i, new_pos in enumerate(pos_list):
        if FLASH_GRAB:
            for pie in dots[i]:
                pie.pos = new_pos
        else:
            dots[i].pos = new_pos


def move_markers(distance):
    pos_list = [
        [
            stim_config["mark_pos"][0] - distance,
            stim_config["mark_pos"][1] + stim_config["center_offset_y"],
        ],
        [
            stim_config["mark_pos"][0] + distance,
            stim_config["mark_pos"][1] - stim_config["center_offset_y"],
        ],
    ]

    for i, new_pos in enumerate(pos_list):
        if FLASH_GRAB:
            for pie in markers[i]:
                pie.pos = new_pos
        else:
            markers[i].pos = new_pos


def shift_frames_by_clock(
    dir_factor,
    time_elapsed,
    frame_default_pos,
):
    """Move the frame along a triangular trajectory with endpoint pauses."""
    one_way_duration = (
        exp["reversal_time"] + exp["dot_duration"]
    )
    n_one_way = int(time_elapsed / one_way_duration)

    time_left_to_shift = time_elapsed % one_way_duration

    if time_left_to_shift > exp["reversal_time"]:
        time_left_to_shift = exp["reversal_time"]

    if n_one_way % 2 == 1:
        time_left_to_shift = (
            exp["reversal_time"] - time_left_to_shift
        )

    shift_by_time = (
        dir_factor
        * stim_config["shift_pix_per_second"]
        * time_left_to_shift
    )

    frame_outer.pos = (
        frame_default_pos[0] + shift_by_time,
        frame_default_pos[1],
    )
    frame_inner.pos = frame_outer.pos


# EYELINK / SCREEN HELPERS

def clear_screen(win):
    event.clearEvents()
    win.fillColor = genv.getBackgroundColor()

    hide_all_dots(markers)
    hide_all_dots(dots)

    fix_outer.autoDraw = False
    fix_inner.autoDraw = False
    frame_outer.autoDraw = False
    frame_inner.autoDraw = False

    win.flip()


def show_msg(win, text, wait_for_keypress=True):
    msg = visual.TextStim(
        win,
        text,
        color=genv.getForegroundColor(),
        wrapWidth=scn_width / 2,
    )

    clear_screen(win)
    msg.draw()
    win.flip()

    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def abort_trial():
    if el_tracker.isRecording():
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    clear_screen(expWin)

    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage(
        "!V CLEAR %d %d %d" % bgcolor_RGB
    )
    el_tracker.sendMessage(
        "TRIAL_RESULT %d" % pylink.TRIAL_ERROR
    )

    return pylink.TRIAL_ERROR


def terminate_task():
    """Save attention data and transfer the EDF before quitting."""

    if el_tracker.isConnected():
        error = el_tracker.isRecording()

        if error == pylink.TRIAL_OK:
            abort_trial()

        el_tracker.setOfflineMode()
        el_tracker.sendCommand("clear_screen 0")
        pylink.msecDelay(500)
        el_tracker.closeDataFile()

        if EYELINK_CONNECTED:
            show_msg(
                expWin,
                "EDF data is transferring from EyeLink Host PC...",
                wait_for_keypress=False,
            )

            try:
                el_tracker.receiveDataFile(
                    edf_file,
                    str(local_edf),
                )
            except RuntimeError as error:
                print("ERROR:", error)

        el_tracker.close()

    expWin.close()
    core.quit()
    sys.exit()


def log_with_time():
    now = datetime.now()
    timestamp = (
        now.strftime("%Y-%m-%d %H:%M:%S.")
        + f"{now.microsecond // 1000:03d}"
    )
    logFile.write(f"[{timestamp}] ")


# KAY RETINOTOPY FIXATION-COLOR ATTENTION TASK

fixation_task = None

def _build_fixation_block_schedule(start_run_s=0.0):
    """Schedule one fixation change in approximately half of all scan blocks."""
    blocks = []
    run_s = float(start_run_s)

    def add_block(block_type, duration_s, trial_number=""):
        nonlocal run_s
        duration_s = float(duration_s)

        if duration_s <= 0:
            raise ValueError(
                f"Nonpositive duration for {block_type}: {duration_s}"
            )

        blocks.append({
            "block_number": len(blocks) + 1,
            "block_type": block_type,
            "trial_number": trial_number,
            "start_run_s": run_s,
            "end_run_s": run_s + duration_s,
            "has_color_change": False,
            "scheduled_time_s": "",
        })

        run_s += duration_s

    # Construct the full run as individual blocks.
    add_block("pre_baseline", exp["pre_run_baseline"])

    for index, trial in enumerate(TRIALS):
        add_block(
            "trial",
            stim_config["duration_stimulus"],
            trial["Trial"],
        )

        if index < len(TRIALS) - 1:
            add_block("ITI", exp["ITI"], trial["Trial"])

    add_block("post_baseline", exp["post_run_baseline"])

    # Randomly select approximately half of all blocks.
    fraction = float(cfg.FIXATION_BLOCK_CHANGE_FRACTION)

    if not 0 <= fraction <= 1:
        raise ValueError(
            "FIXATION_BLOCK_CHANGE_FRACTION must be between 0 and 1"
        )

    n_selected = int(np.floor(len(blocks) * fraction + 0.5))

    selected_indices = set(
        np.random.choice(
            len(blocks),
            size=n_selected,
            replace=False,
        ).tolist()
    )

    # Randomize one change time within each selected block.
    hz = float(cfg.FIXATION_TASK_HZ)
    margin_s = float(cfg.FIXATION_CHANGE_EDGE_MARGIN_S)

    if hz <= 0 or margin_s < 0:
        raise ValueError(
            "Fixation task Hz must be positive and margin nonnegative"
        )

    for index, block in enumerate(blocks):
        if index not in selected_indices:
            continue

        first_frame = int(
            np.ceil((block["start_run_s"] + margin_s) * hz - 1e-8)
        )
        last_frame = int(
            np.floor((block["end_run_s"] - margin_s) * hz + 1e-8)
        )

        if first_frame > last_frame:
            raise ValueError(
                f"Block {block['block_number']} is too short for the margin"
            )

        block["has_color_change"] = True
        block["scheduled_time_s"] = (
            int(np.random.randint(first_frame, last_frame + 1)) / hz
        )

    return blocks


def _set_fixation_color(index):
    color = cfg.FIXATION_COLORS_RGB255[index]
    fix_outer.fillColor = color
    fix_outer.lineColor = color


def _record_fixation_event(event):
    if fixation_task is not None:
        fixation_task["event_log"].append(event)

def start_fixation_task(start_run_s=0.0):
    """Start red fixation and randomize the entire scan's block schedule."""
    global fixation_task

    _set_fixation_color(0)

    blocks = _build_fixation_block_schedule(start_run_s)

    change_schedule = [
        block for block in blocks
        if block["has_color_change"]
    ]

    fixation_task = {
        "current_color_index": 0,
        "change_schedule": change_schedule,
        "next_change_index": 0,
        "change_count": 0,
        "event_log": [],
    }

    _record_fixation_event(
        {
            "event_type": "task_start",
            "run_time_s": float(start_run_s),
            "scheduled_time_s": float(start_run_s),
            "color": cfg.FIXATION_COLOR_NAMES[0],
            "key": "",
        }
    )


def update_fixation_task(run_time_s):
    """Apply each selected block's single scheduled color change."""
    if fixation_task is None:
        return

    schedule = fixation_task["change_schedule"]

    while fixation_task["next_change_index"] < len(schedule):
        block = schedule[fixation_task["next_change_index"]]
        scheduled = block["scheduled_time_s"]

        if run_time_s < scheduled:
            break

        fixation_task["next_change_index"] += 1

        # Never carry a missed color change into a later block.
        if run_time_s >= block["end_run_s"]:
            _record_fixation_event({
                "event_type": "missed_color_change",
                "run_time_s": float(run_time_s),
                "scheduled_time_s": float(scheduled),
                "color": cfg.FIXATION_COLOR_NAMES[
                    fixation_task["current_color_index"]
                ],
                "key": "",
            })

            logFile.write(
                f"WARNING: missed fixation change in block "
                f"{block['block_number']} ({block['block_type']})\n"
            )
            continue

        # Select a color different from the current color.
        current = fixation_task["current_color_index"]

        available = [
            idx for idx in range(len(cfg.FIXATION_COLORS_RGB255))
            if idx != current
        ]

        new_index = int(np.random.choice(available))

        fixation_task["current_color_index"] = new_index
        fixation_task["change_count"] += 1

        _set_fixation_color(new_index)

        _record_fixation_event({
            "event_type": "color_change",
            "run_time_s": float(run_time_s),
            "scheduled_time_s": float(scheduled),
            "color": cfg.FIXATION_COLOR_NAMES[new_index],
            "key": "",
        })

        el_tracker.sendMessage(
            f"FIXATION_COLOR_CHANGE "
            f"{fixation_task['change_count']} "
            f"{cfg.FIXATION_COLOR_NAMES[new_index]}"
        )


def collect_fixation_task_keys():
    """Log fixation-task responses and return whether escape was pressed."""
    if fixation_task is None:
        return False

    keys = event.getKeys(
        keyList=cfg.FIXATION_RESPONSE_KEYS + ["escape"],
        timeStamped=exp_clock,
    )

    escaped = False

    for key, key_time in keys:
        if key == "escape":
            escaped = True
            continue

        _record_fixation_event(
            {
                "event_type": "button_press",
                "run_time_s": float(key_time),
                "scheduled_time_s": np.nan,
                "color": cfg.FIXATION_COLOR_NAMES[
                    fixation_task["current_color_index"]
                ],
                "key": key,
            }
        )

        el_tracker.sendMessage(f"FIXATION_BUTTON {key}")

    return escaped

def bids_trial_type(trial):
    """Identify all seven scan conditions."""
    condition = trial["ExpCondition"]

    if condition == cfg.COND_VERIDICAL:
        return "veridical"
    if condition == cfg.COND_FIPS:
        return f"fips_{trial['InducedPercept']}"
    if condition == cfg.COND_PERCEPTUAL:
        return f"perceptual_{trial['InducedPercept']}"
    if condition == cfg.COND_FRAME_ONLY:
        return f"frame_only_starts_{trial['FrameStartPosition']}"

    raise ValueError(f"Unknown scan condition: {condition!r}")


# VALIDATE SCAN TRIAL COUNTS

if expInfo["ses"] == "scan" and not CODE_TEST:
    run_type = TRIALS[0]["RunType"]

    if run_type == "A":
        expected = cfg.RUN_A_TRIAL_COUNTS
    elif run_type == "B":
        expected = cfg.RUN_B_TRIAL_COUNTS
    else:
        raise ValueError(f"Unknown run type: {run_type}")

    actual = {condition: 0 for condition in expected}

    for trial in TRIALS:
        if trial["RunType"] != run_type:
            raise ValueError("Mixed run types detected")

        condition = bids_trial_type(trial)

        if condition not in actual:
            raise ValueError(
                f"Unexpected condition in Run {run_type}: {condition}"
            )

        actual[condition] += 1

    if actual != expected:
        raise ValueError(
            f"Run {run_type} trial counts do not match.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )


def save_bids_events(run_end_s):
    """Write one events.tsv row per trial, ITI, and baseline."""
    if BIDS_EVENTS_NAME is None or not dataFile or fixation_task is None:
        return

    events = fixation_task["event_log"]

    # Actual displayed changes, not merely scheduled changes.
    changes = [
        float(e["run_time_s"]) for e in events
        if e["event_type"] == "color_change"
    ]
    presses = [
        float(e["run_time_s"]) for e in events
        if e["event_type"] == "button_press"
    ]

    response_window = float(cfg.FIXATION_RESPONSE_WINDOW_S)
    if response_window <= 0:
        raise ValueError("FIXATION_RESPONSE_WINDOW_S must be positive")

    rows = []

    def add_row(start, end, kind):
        start, end = float(start), float(end)

        if end < start:
            raise ValueError(f"Negative {kind} duration: {start} to {end}")

        block_changes = [t for t in changes if start <= t < end]

        if block_changes:
            correct = all(
                any(
                    t <= p <= min(t + response_window, end)
                    for p in presses
                )
                for t in block_changes
            )
            accuracy = "true" if correct else "false"
        else:
            accuracy = "n/a"

        rows.append({
            "onset": round(start, 6),
            "duration": round(end - start, 6),
            "trial_type": kind,
            "fixation_change": "true" if block_changes else "false",
            "accuracy": accuracy,
        })

    first = dataFile[0]
    add_row(0.0, first["stim_onset_run_s"], "baseline")

    for index, trial in enumerate(dataFile):
        onset = trial["stim_onset_run_s"]
        offset = trial["stim_offset_run_s"]

        add_row(onset, offset, bids_trial_type(trial))

        if index + 1 < len(dataFile):
            next_onset = dataFile[index + 1]["stim_onset_run_s"]
            add_row(offset, next_onset, "iti")

    add_row(
        dataFile[-1]["stim_offset_run_s"],
        run_end_s,
        "baseline"
    )

    pd.DataFrame(rows, columns=[
        "onset", "duration", "trial_type",
        "fixation_change", "accuracy"
    ]).to_csv(
        BIDS_EVENTS_NAME,
        sep="\t",
        index=False,
        na_rep="n/a"
    )

    logFile.write(
        f"Saved {len(rows)} BIDS events: {BIDS_EVENTS_NAME}\n"
    )


def prepare_fixation_only_screen():
    """Hide the frame, probes, and comparison markers."""
    hide_all_dots(dots)
    hide_all_dots(markers)

    frame_outer.autoDraw = False
    frame_inner.autoDraw = False

    fix_outer.autoDraw = True
    fix_inner.autoDraw = True


def run_fixation_until(target_run_s):
    """Run the fixation-color task until the scheduled time."""
    prepare_fixation_only_screen()

    while exp_clock.getTime() < target_run_s:
        now = exp_clock.getTime()
        update_fixation_task(now)

        expWin.flip()

        if collect_fixation_task_keys():
            terminate_task()


def check_fixation(
    fix_duration=None,
    fix_radius_dva=None,
    eye_used=None,
):
    """Require continuous gaze within the rectangular fixation gate."""
    if fix_duration is None:
        fix_duration = cfg.FIXATION_GATE_DURATION_S
    if fix_radius_dva is None:
        fix_radius_dva = cfg.FIXATION_GATE_RADIUS_DVA
    if eye_used is None:
        eye_used = cfg.EYE_USED

    fix_x, fix_y = stim_config["fix_pos"]
    fixation_radius_px = fix_radius_dva * ppd

    fix_outer.autoDraw = True
    fix_inner.autoDraw = True

    fixation_validated = False
    gaze_start_time = None
    old_sample = None

    expWin.flip()
    fix_clock = core.Clock()

    while not fixation_validated:
        new_sample = el_tracker.getNewestSample()

        if new_sample is None:
            continue

        if (
            old_sample is not None
            and new_sample.getTime() == old_sample.getTime()
        ):
            continue

        old_sample = new_sample

        if eye_used == 1 and new_sample.isRightSample():
            g_x, g_y = new_sample.getRightEye().getGaze()
        elif eye_used == 0 and new_sample.isLeftSample():
            g_x, g_y = new_sample.getLeftEye().getGaze()
        else:
            continue

        if g_x is None or g_y is None:
            continue

        gx_centered = g_x - scn_width / 2
        gy_centered = (g_y - scn_height / 2) * -1

        # Fixation uses a rectangular acceptance region.
        inside = (
            abs(gx_centered - fix_x) < fixation_radius_px
            and abs(gy_centered - fix_y) < fixation_radius_px
        )

        if inside:
            if gaze_start_time is None:
                gaze_start_time = fix_clock.getTime()
                fix_outer.fillColor = cfg.FIXATION_COLORS_RGB255[0]
                fix_outer.lineColor = cfg.FIXATION_COLORS_RGB255[0]
                expWin.flip()

            elif (
                fix_clock.getTime() - gaze_start_time
                >= fix_duration
            ):
                el_tracker.sendMessage("FIXATION_ACHIEVED")
                fixation_validated = True

        else:
            gaze_start_time = None
            fix_outer.fillColor = "red"
            fix_outer.lineColor = "red"
            expWin.flip()

    fix_outer.fillColor = cfg.FIXATION_COLORS_RGB255[0]
    fix_outer.lineColor = cfg.FIXATION_COLORS_RGB255[0]


# SCAN TRIAL

def show_stimulus(
    trial,
    when_to_flip,
    add_iti_after=True,
):
    """Present one scan trial and optionally its fixation-only ITI."""

    trial_type = trial["ExpCondition"]
    dir_map = {"left": -1, "right": 1, "NA": 0}
    dir_factor = dir_map.get(
        trial["InitialDirection"],
        0,
    )

    dot_pos = trial["InitialPosition"]
    dot_color = (
        "black" if np.random.randint(0, 2) == 1 else "white"
    )

    frame_present = trial_type in {
        cfg.COND_FIPS,
        cfg.COND_FRAME_ONLY,
    }

    probes_present = trial_type != cfg.COND_FRAME_ONLY

    perceived_distance = 0.0

    if trial_type == cfg.COND_PERCEPTUAL:
        if trial["InducedPercept"] == "left":
            perceived_distance = deg2pix(PRESCAN_LEFT)

        elif trial["InducedPercept"] == "right":
            perceived_distance = -deg2pix(PRESCAN_RIGHT)

        if (
            PRESCAN_LEFT == 0.0
            and PRESCAN_RIGHT == 0.0
        ):
            raise RuntimeError(
                "Perceptual condition requested but no prescan displacement was loaded. "
                f"Run prescan first or provide {predataName}. "
                "For testing, use a dummy scan (Eyelink Connected: no) "
                "or set PREVIEW_MODE = True in fips_config.py."
            )

    frame_default_pos = None

    if frame_present:
        frame_default_pos = (
            stim_config["frame_center_x"]
            - 0.5 * stim_config["frame_travel"] * dir_factor,
            stim_config["stim_pos"][1],
        )
        frame_outer.pos = frame_default_pos
        frame_inner.pos = frame_default_pos
        frame_outer.autoDraw = True
        frame_inner.autoDraw = True

    else:
        frame_outer.autoDraw = False
        frame_inner.autoDraw = False

    if trial_type in {
        cfg.COND_FIPS,
        cfg.COND_VERIDICAL,
    }:
        move_dots(0)

    elif trial_type == cfg.COND_PERCEPTUAL:
        move_dots(perceived_distance)

    elif trial_type == cfg.COND_FRAME_ONLY:
        hide_all_dots()

    if (
        cfg.REQUIRE_FIXATION_BEFORE_TRIAL
        and EYELINK_CONNECTED
    ):
        check_fixation()

    if expInfo["ses"] == "scan":
        run_fixation_until(when_to_flip)
    else:
        wait_duration = when_to_flip - exp_clock.getTime()
        if wait_duration > 0:
            core.wait(wait_duration)

    # Restore the frame after the fixation-only wait disables it.
    if frame_present:
        frame_outer.pos = frame_default_pos
        frame_inner.pos = frame_default_pos
        frame_outer.autoDraw = True
        frame_inner.autoDraw = True
    else:
        frame_outer.autoDraw = False
        frame_inner.autoDraw = False

    scheduled_onset = when_to_flip
    scheduled_offset = (
        scheduled_onset + stim_config["duration_stimulus"]
    )

    next_onset = scheduled_offset

    if add_iti_after:
        next_onset += exp["ITI"]

    nt_label = trial["Trial"]

    el_tracker.sendMessage(f"TRIALID {nt_label}")
    el_tracker.sendCommand(
        f"record_status_message 'TRIAL number {nt_label}'"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR condition {trial_type}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR induced_percept {trial['InducedPercept']}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR initial_position {trial['InitialPosition']}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR initial_direction {trial['InitialDirection']}"
    )

    log_with_time()
    logFile.write(
        f"Trial {nt_label} Start; "
        f"Condition={trial_type}\n"
    )

    trial["scheduled_onset_run_s"] = scheduled_onset
    trial["scheduled_offset_run_s"] = scheduled_offset

    # Present the scan stimulus for an exact number of frames.
    refresh_hz = float(monInfo["refresh_rate"])
    exact_frames = stim_config["duration_stimulus"] * refresh_hz
    n_frames = round(exact_frames)

    if abs(exact_frames - n_frames) > 1e-6:
        raise ValueError(
            "Stimulus duration must correspond to a whole "
            f"number of frames. Got {exact_frames}."
        )

    def stamp_flip(field):
        """Record the display flip relative to scanner trigger."""
        trial[field] = exp_clock.getTime()

    expWin.recordFrameIntervals = True
    dot_counter = 0

    for frame_index in range(n_frames):
        # Use the center of each frame interval for flash timing.
        # At 60 Hz: 27 movement frames + 3 flash frames.
        flash_time_s = (frame_index + 0.5) / refresh_hz
        motion_time_s = frame_index / refresh_hz

        update_fixation_task(exp_clock.getTime())

        if probes_present:
            _, dot_counter = update_dots_by_clock(
                flash_time_s,
                dot_color,
                dot_counter,
            )
        else:
            hide_all_dots()

        if frame_present:
            shift_frames_by_clock(
                dir_factor,
                motion_time_s,
                frame_default_pos,
            )

        if frame_index == 0:
            expWin.callOnFlip(
                stamp_flip, "stim_onset_run_s"
            )
            expWin.callOnFlip(
                el_tracker.sendMessage,
                f"STIM_ON {nt_label}",
            )

        expWin.flip()

        if collect_fixation_task_keys():
            terminate_task()

    # Switch to fixation only after 600 stimulus frames.
    prepare_fixation_only_screen()

    expWin.callOnFlip(
        stamp_flip, "stim_offset_run_s"
    )
    expWin.flip()

    expWin.recordFrameIntervals = False



    el_tracker.sendMessage(f"STIM_OFF {nt_label}")
    el_tracker.sendMessage(f"TRIAL END {nt_label}")
    el_tracker.sendMessage(
        "TRIAL_RESULT %d" % pylink.TRIAL_OK
    )

    log_with_time()
    logFile.write(
        f"Trial {nt_label} Stimulus Off\n"
    )

    return trial, next_onset


# PRESCAN MATCHING

def test_stimulus(trial):
    """Alternate FIPS and adjustable physical-displacement comparison until accepted."""
    trial_type = cfg.COND_FIPS

    dir_map = {"left": -1, "right": 1}
    dir_factor = dir_map[trial["InitialDirection"]]

    dot_pos = trial["InitialPosition"]
    dot_color = "white"
    dot_counter = 0

    marker_dist = np.random.uniform(0, 1 * ppd)
    move_dots(0)
    move_markers(marker_dist)

    Resp = False

    nt_label = trial["Trial"]

    el_tracker.sendMessage(f"TRIALID {nt_label}")
    el_tracker.sendCommand(
        f"record_status_message 'TRIAL number {nt_label}'"
    )

    el_tracker.sendMessage(
        f"!V TRIAL_VAR condition {trial['ExpCondition']}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR induced_percept {trial['InducedPercept']}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR initial_position {trial['InitialPosition']}"
    )
    el_tracker.sendMessage(
        f"!V TRIAL_VAR initial_direction {trial['InitialDirection']}"
    )

    log_with_time()
    logFile.write(
        f"Prescan Trial {nt_label} Start\n"
    )

    expWin.recordFrameIntervals = True
    trial_clock = core.Clock()
    trial_clock.reset()

    t_duration = stim_config["duration_stimulus"]
    onset_sent = False

    while not Resp:
        if trial_type == cfg.COND_FIPS:
            frame_default_pos = (
                stim_config["frame_center_x"]
                - 0.5
                * stim_config["frame_travel"]
                * dir_factor,
                stim_config["stim_pos"][1],
            )

            frame_outer.pos = frame_default_pos
            frame_inner.pos = frame_default_pos
            frame_outer.autoDraw = True
            frame_inner.autoDraw = True

            hide_all_dots(markers)

        else:
            frame_outer.autoDraw = False
            frame_inner.autoDraw = False
            hide_all_dots(dots)

        Show = True

        while Show:
            _, wheel_dY = mouseResp.getWheelRel()
            marker_dist += wheel_dY * 2

            time_elapsed = trial_clock.getTime()

            if trial_type == cfg.COND_FIPS:
                shift_frames_by_clock(
                    dir_factor,
                    time_elapsed,
                    frame_default_pos,
                )

                dot_color, dot_counter = update_dots_by_clock(
                    time_elapsed,
                    dot_color,
                    dot_counter,
                )

            else:
                move_markers(marker_dist)
                dot_color = update_markers_by_clock(
                    time_elapsed,
                    dot_color,
                    dot_pos,
                )

            if not onset_sent:
                expWin.callOnFlip(
                    el_tracker.sendMessage,
                    f"STIM_ON {nt_label}",
                )

            expWin.flip()
            onset_sent = True

            all_keys = event.getKeys()

            if "space" in all_keys or "4" in all_keys:
                trial["response_pix"] = marker_dist
                trial["response_dva"] = marker_dist / ppd
                trial["response_time"] = trial_clock.getTime()

                el_tracker.sendMessage(
                    f"TRIAL {nt_label}: response_saved"
                )

                log_with_time()
                logFile.write(
                    f"Prescan Trial {nt_label} Response Saved\n"
                )

                Resp = True
                Show = False

            elif "1" in all_keys:
                marker_dist += ppd / 10

            elif "2" in all_keys:
                marker_dist -= ppd / 10

            elif (
                "escape" in all_keys
                and trial_clock.getTime() > 1
            ):
                terminate_task()

            if trial_clock.getTime() > t_duration:
                trial_type = (
                    cfg.COND_PERCEPTUAL
                    if trial_type == cfg.COND_FIPS
                    else cfg.COND_FIPS
                )

                t_duration += stim_config["duration_stimulus"]
                log_with_time()
                logFile.write(
                    f"Prescan Trial {nt_label} "
                    f"Switched to {trial_type}\n"
                )
                Show = False

    event.clearEvents()
    hide_all_dots()
    hide_all_dots(markers)

    frame_outer.autoDraw = False
    frame_inner.autoDraw = False

    fix_outer.color = "green"
    expWin.recordFrameIntervals = False
    expWin.flip()

    core.wait(0.2)

    fix_outer.color = cfg.FIXATION_COLORS_RGB255[0]
    expWin.flip()

    el_tracker.sendMessage(f"STIM_OFF {nt_label}")
    el_tracker.sendMessage(f"TRIAL END {nt_label}")
    el_tracker.sendMessage(
        "TRIAL_RESULT %d" % pylink.TRIAL_OK
    )

    return trial


# RUN

if expInfo["ses"] == "scan":
    run_type = TRIALS[0]["RunType"]

    if run_type == "B" and not use_fake_prescan and not predataName.exists():
        show_msg(
            expWin,
            "ERROR: Run B requires prescan measurements.\n\n"
            "Complete the participant's prescan before continuing.",
            True,
        )
        terminate_task()

    # Screen 2: Experiment information.
    show_msg(
        expWin,
        f"FIPS Run {expInfo['run']} — Type {run_type}\n"
        f"Trials: {len(TRIALS)}\n"
        f"Duration: {planned_run_s:.0f} seconds "
        f"({planned_run_s / cfg.SCANNER_TR_S:g} TRs)\n\n"
        + task_text,
        True,
    )

    # Preserve prescan information for real EyeLink scans.
    if not DUMMY_SCAN:
        if use_fake_prescan or predataName.exists():
            show_msg(expWin, prescan_text, True)


# EyeLink calibration

if EYELINK_CONNECTED:
    eyelink_msg = (
        "Calibrate the tracker.\n\n"
        "Please look at the center of each target "
        "and stay still until the next one appears."
    )
    show_msg(expWin, eyelink_msg)

    try:
        el_tracker.doTrackerSetup()
        el_tracker.setOfflineMode()
    except RuntimeError as err:
        print("ERROR:", err)
        el_tracker.exitCalibration()


# Scanner trigger

if DUMMY_SCAN:
    # Screen 3: Dummy prescan values.
    # Pressing 5 here starts the experiment directly.
    show_msg(
        expWin,
        prescan_text
        + f"\n\nPress {cfg.SCANNER_TRIGGER_KEY} to start.",
        wait_for_keypress=False,
    )

    # Start dummy EyeLink recording before the trigger.
    el_tracker.startRecording(1, 1, 1, 1)

elif expInfo["ses"] == "scan":
    # Real scanner: preserve fixation before the trigger.
    _set_fixation_color(0)
    prepare_fixation_only_screen()
    expWin.color = stim_config["backColor"]
    expWin.flip(clearBuffer=True)

    el_tracker.startRecording(1, 1, 1, 1)

    print(
        f"Waiting for scanner signal "
        f"({cfg.SCANNER_TRIGGER_KEY})..."
    )

else:
    # Preserve prescan behavior.
    show_msg(
        expWin,
        f"Waiting for scanner signal "
        f"({cfg.SCANNER_TRIGGER_KEY})...",
        False,
    )

event.clearEvents()
event.waitKeys(keyList=[cfg.SCANNER_TRIGGER_KEY])

if DUMMY_SCAN:
    # Replace the dummy prescan screen with fixation.
    _set_fixation_color(0)
    prepare_fixation_only_screen()
    expWin.color = stim_config["backColor"]
    expWin.flip(clearBuffer=True)

# The scanner trigger establishes time zero.
exp_clock = core.Clock()
logging.setDefaultClock(exp_clock)
t_start = exp_clock.getTime()

log_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)
logFile.write(
    f"Run {expInfo['run']} started at {log_time}\n"
)

dataFile = []

if expInfo["ses"] != "scan":
    prepare_fixation_only_screen()
    expWin.color = stim_config["backColor"]
    expWin.flip(clearBuffer=True)
    el_tracker.startRecording(1, 1, 1, 1)

el_tracker.sendMessage("RUN_START")


if expInfo["ses"] == "scan":
    el_tracker.sendMessage(
        f"RUN_TYPE {TRIALS[0]['RunType']}"
    )
    start_fixation_task(start_run_s=0.0)


when_to_flip = exp["pre_run_baseline"]

logFile.write(
    f"Pre-run fixation baseline = "
    f"{exp['pre_run_baseline']:.3f} s\n"
)


# Trial loop

for trial_index, trial in enumerate(TRIALS):
    tt_start = exp_clock.getTime()

    if expInfo["ses"] == "scan":
        is_last_trial = (
            trial_index == len(TRIALS) - 1
        )

        trial, when_to_flip = show_stimulus(
            trial,
            when_to_flip,
            add_iti_after=not is_last_trial,
        )

    else:
        test_stimulus(trial)

    trial_duration = exp_clock.getTime() - tt_start
    logFile.write(
        f"Trial Duration: {trial_duration:.3f} seconds\n"
    )

    executed_trial = trial.copy()
    dataFile.append(executed_trial)

    if expInfo["ses"] == "prescan":
        prescan_progress = pd.DataFrame(dataFile)

        # Summary values remain blank until all trials finish.
        prescan_progress.insert(0, "avg", "")
        prescan_progress.insert(0, "right", "")
        prescan_progress.insert(0, "left", "")

        prescan_progress.to_csv(
            predataName,
            index=False,
        )

# Post-run baseline

if expInfo["ses"] == "scan":
    post_baseline_end = (
        max(when_to_flip, dataFile[-1]["stim_offset_run_s"])
        + exp["post_run_baseline"]
    )


    run_fixation_until(post_baseline_end)
    save_bids_events(run_end_s=exp_clock.getTime())

    el_tracker.sendMessage("RUN_END")

    logFile.write(
        f"Post-run fixation baseline = "
        f"{exp['post_run_baseline']:.3f} s\n"
    )
else:
    el_tracker.sendMessage("RUN_END")


# WRAP UP

t_end = exp_clock.getTime()
exp_duration = t_end - t_start

logFile.write(
    f"The experiment took {exp_duration:.3f} seconds\n"
)

if expInfo["ses"] == "scan":
    current_run = int(expInfo["run"])
    end_text = (
        end_end_text
        if current_run == exp["n_runs"]
        else end_run_text
    )

    show_msg(expWin, end_text)
    core.wait(0.2)


elif expInfo["ses"] == "prescan":
    prescan_df = pd.DataFrame(dataFile)

    grouped = prescan_df.groupby(
        "InducedPercept"
    )["response_dva"].mean()

    left_mean = abs(
        grouped.get("left", float("nan"))
    )
    right_mean = abs(
        grouped.get("right", float("nan"))
    )
    avg_abs = (left_mean + right_mean) / 2

    # Preserve all individual responses alongside the summary values.
    prescan_summary = prescan_df.copy()

    prescan_summary.insert(0, "avg", avg_abs)
    prescan_summary.insert(0, "right", right_mean)
    prescan_summary.insert(0, "left", left_mean)

    prescan_summary.to_csv(
        predataName,
        index=False,
    )

    result_text = (
        f"Left mean: {left_mean:.3f} dva\n"
        f"Right mean: {right_mean:.3f} dva\n"
        f"Average: {avg_abs:.3f} dva"
    )

    show_msg(
        expWin,
        result_text,
        True,
    )


if el_tracker.isRecording():
    pylink.pumpDelay(100)
    el_tracker.stopRecording()

terminate_task()

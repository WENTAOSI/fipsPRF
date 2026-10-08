#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-hemifield FIPS thesis experiment.

This file deliberately keeps the architecture and scanner/EyeLink workflow
close to WENTAOSI/fipsPRF/exp/code/fips_stim.py.

What is retained from fips_stim.py:
    - PsychoPy GUI and monitor selection
    - subject/run bookkeeping and CSV logging
    - automatic loading of the participant's prescan FIPS magnitude
    - EyeLink connection / dummy mode
    - EDF creation, calibration, recording, messages, transfer
    - scanner trigger key "5"
    - frame-interval logging
    - prescan matching workflow
    - 450 ms frame motion + 50 ms endpoint flash logic
    - checkerboard probes
    - graceful end-of-run handling

What changes for the thesis:
    - one visual hemifield, controlled in fips_config.py
    - four scan conditions:
          FIPS
          Veridical
          Perceptual
          FrameOnly
    - 28 trials/run: 8 FIPS, 4 Veridical, 8 Perceptual, and 8 FrameOnly, in 3 mini-blocks
    - 15 s pre/post fixation-only baselines and 6 s fixation-only ITIs
    - frame-only condition
    - all tunable experiment parameters moved to fips_config.py
    - exact Kay-style fixation-color attention task during scan periods:
      4-pixel dot, 50% alpha, red/black/white, random ~1-5 s changes,
      participant presses a response button whenever the color changes

Keep this file in the same directory as:
    fips_config.py
    EyeLinkCoreGraphicsPsychoPy.py
"""

# ===========================================================================
# DEBUG / IMPORTS
# ===========================================================================

from psychopy import visual, event, core, monitors, logging, gui, data
from psychopy.tools.attributetools import undefined
import numpy as np
import copy
import os
import pandas as pd
import csv
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
    """Request the same real display mode used by MATLAB pton/SetResolution."""
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
    """Use a real fullscreen display mode and restore it when closing."""

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
                # Scissor coordinates use framebuffer pixels; PsychoPy's pix
                # drawing units use half that resolution on Retina displays.
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


# ===========================================================================
# SESSION INFO / DATA PATHS
# ===========================================================================

expInfo = {
    "sub_id": cfg.DEFAULT_SUB_ID,
    "sub_name": cfg.DEFAULT_SUB_NAME,
    "run": cfg.DEFAULT_RUN,
}

ROOTDIR = Path(__file__).resolve().parent
DATADIR = (ROOTDIR / cfg.DATA_DIR).resolve()
DATADIR.mkdir(parents=True, exist_ok=True)

# Preserve the source script's convenience behavior, but only use actual
# trial-data CSVs when deciding what subject/run was most recent.
sub_dirs = sorted(
    [d for d in DATADIR.iterdir() if d.is_dir() and d.name.startswith("sub-")],
    key=os.path.getctime,
    reverse=True,
)

if sub_dirs:
    recent_subdir = sub_dirs[0]
    sub_id_guess = recent_subdir.name.replace("sub-", "")
    csv_files = sorted(
        recent_subdir.glob("*_data.csv"),
        key=os.path.getctime,
        reverse=True,
    )

    if csv_files:
        recent_file = csv_files[0]
        name_parts = recent_file.stem.split("_")
        # Expected:
        # sub-01_AB_FIPS_scan_3_data
        if len(name_parts) >= 6:
            file_sub_name = name_parts[1]
            file_session = name_parts[3]
            file_run = name_parts[4]

            if file_run.isdigit():
                expInfo["sub_id"] = sub_id_guess.zfill(2)
                expInfo["sub_name"] = file_sub_name

                if file_session == "prescan":
                    expInfo["run"] = file_run
                elif file_session == "scan":
                    last_run = int(file_run)
                    if last_run < cfg.SCAN["n_runs"]:
                        expInfo["run"] = str(last_run + 1)
                    else:
                        expInfo["run"] = str(cfg.SCAN["n_runs"])


# ===========================================================================
# GUI
# ===========================================================================

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

FLASH_GRAB = expInfo["stimulus"] == "checkerboard"
EYELINK_CONNECTED = expInfo["eyelink"] == "yes"

# For this experiment, a dummy scan means:
#   Session = scan
#   Eyelink Connected = no
#
# Dummy scans are for testing only and can use fake prescan displacement
# values so that Perceptual trials render without a participant prescan.
DUMMY_SCAN = (
    expInfo["ses"] == "scan"
    and not EYELINK_CONNECTED
)


# ===========================================================================
# SAVING / LOGGING
# ===========================================================================

sub_id = f"{int(expInfo['sub_id']):02d}"
run_id = f"{int(expInfo['run']):02d}"

SUBDIR = DATADIR / f"sub-{sub_id}"
LOGDIR = DATADIR / "logs"
EDFDIR = DATADIR / "edf"

for directory in [SUBDIR, LOGDIR, EDFDIR]:
    directory.mkdir(parents=True, exist_ok=True)

trialsName = (
    SUBDIR
    / f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_{expInfo['ses']}_{expInfo['run']}_psychophysics.csv"
)
dataName = (
    SUBDIR
    / f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_{expInfo['ses']}_{expInfo['run']}_data.csv"
)

framesName = (
    LOGDIR
    / f"sub-{sub_id}_{TASK}_{expInfo['ses']}_{expInfo['run']}_frames.csv"
)
fixationTaskName = (
    LOGDIR
    / f"sub-{sub_id}_{TASK}_{expInfo['ses']}_{expInfo['run']}_fixation_task.csv"
)
logName = (
    LOGDIR
    / f"sub-{sub_id}_{TASK}_{expInfo['ses']}_{expInfo['run']}.log"
)

logFile = logging.LogFile(str(logName), level=logging.EXP)
logFile.write(
    f"ID: {expInfo['sub_id']}, "
    f"Subject: {expInfo['sub_name']}, "
    f"Experiment: {TASK}, "
    f"Session: {expInfo['ses']}, "
    f"Run: {expInfo['run']}\n"
)
logging.console.setLevel(logging.WARNING)

# EyeLink Host filenames should stay short. For the default TASK="FIPS":
# e.g. 01FIPS01.EDF
edf_fname = f"{sub_id}{TASK[:4].upper()}{run_id}"
edf_file = edf_fname + ".EDF"

# Unlike the source script, include run in the LOCAL EDF filename so repeated
# scan runs do not overwrite each other.
session_identifier = (
    f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_{expInfo['ses']}_run-{run_id}"
)
session_folder = EDFDIR / session_identifier
session_folder.mkdir(parents=True, exist_ok=True)
local_edf = session_folder / f"{session_identifier}.EDF"


# ===========================================================================
# LOAD PRESCAN SUMMARY
# ===========================================================================

PRESCAN_LEFT = 0.0
PRESCAN_RIGHT = 0.0
PRESCAN_AVG = 0.0

predataName = (
    SUBDIR
    / f"sub-{sub_id}_{expInfo['sub_name']}_{TASK}_prescan_summary.csv"
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

elif predataName.exists():
    prescan_summary_df = pd.read_csv(predataName)
    PRESCAN_LEFT = float(prescan_summary_df.loc[0, "left"])
    PRESCAN_RIGHT = float(prescan_summary_df.loc[0, "right"])
    PRESCAN_AVG = float(prescan_summary_df.loc[0, "avg"])
    prescan_text = (
        "Loaded prescan summary:\n"
        f"Left: {PRESCAN_LEFT:.3f} dva\n"
        f"Right: {PRESCAN_RIGHT:.3f} dva\n"
        f"Average: {PRESCAN_AVG:.3f} dva"
    )

else:
    prescan_text = f"Prescan summary file not found:\n{predataName}"


# ===========================================================================
# MONITOR / WINDOW
# ===========================================================================

monName = expInfo["monitor"]
monInfo = cfg.MONITORS[monName]

expMon = monitors.Monitor(
    name=monName,
    width=monInfo["size_cm"][0],
    distance=monInfo["mon_dist"],
)
expMon.setSizePix(monInfo["size_px"])
expMon.save()

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
# Exact conversion used by the local pRF runretinotopy.m.
cm_per_dva = viewing_distance_cm * 0.017455
# Scale the entire logical 1024x768 image only when the native mode failed.
render_scale = expWin.render_scale
ppd = cm_per_dva * px_per_cm * render_scale
if monName == "Macbook":
    # Match Scanner's composition, independently of laptop physical geometry.
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


# ===========================================================================
# EYELINK SETUP
# ===========================================================================

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


# ===========================================================================
# EXPERIMENT PARAMETERS
# ===========================================================================

exp = copy.deepcopy(
    cfg.SCAN if expInfo["ses"] == "scan" else cfg.PRESCAN
)

if CODE_TEST:
    exp["n_trials"] = cfg.CODE_TEST_N_TRIALS


# ===========================================================================
# TRIAL GENERATION
# ===========================================================================

def source_initial_direction(initial_position, induced_percept):
    """
    Same FIPS mapping used in the source fips_stim.py.
    """
    if (
        (initial_position == "upper" and induced_percept == "left")
        or
        (initial_position == "lower" and induced_percept == "right")
    ):
        return "right"
    return "left"


def eight_fips_trials():
    """Two trials per apparent-tilt / frame-start-position combination."""
    trials = []
    for induced_percept in ("left", "right"):
        for frame_start in ("left", "right"):
            # A frame starting on the left first moves right, and vice versa.
            direction = "right" if frame_start == "left" else "left"
            initial_position = next(
                pos for pos in ("upper", "lower")
                if source_initial_direction(pos, induced_percept) == direction
            )
            for _ in range(2):
                trials.append({
                    "ExpCondition": cfg.COND_FIPS,
                    "InitialPosition": initial_position,
                    "InducedPercept": induced_percept,
                    "InitialDirection": direction,
                    "FrameStartPosition": frame_start,
                    "FIPSCategory": f"{induced_percept}_tilt_frame_starts_{frame_start}",
                })
    return trials


def eight_perceptual_trials():
    """Four trials per tilt, with two upper-first and two lower-first each."""
    return [
        {
            "ExpCondition": cfg.COND_PERCEPTUAL,
            "InitialPosition": position,
            "InducedPercept": tilt,
            "InitialDirection": "NA",
        }
        for tilt in ("left", "right")
        for position in ("upper", "lower")
        for _ in range(2)
    ]


def four_veridical_trials():
    return [
        {
            "ExpCondition": cfg.COND_VERIDICAL,
            "InitialPosition": pos,
            "InducedPercept": "none",
            "InitialDirection": "NA",
        }
        for pos in ["upper", "lower", "upper", "lower"]
    ]


def eight_frame_only_trials():
    # Four trials start on each side, moving toward the opposite side.
    return [
        {
            "ExpCondition": cfg.COND_FRAME_ONLY,
            "InitialPosition": "NA",
            "InducedPercept": "none",
            "InitialDirection": direction,
            "FrameStartPosition": "left" if direction == "right" else "right",
        }
        for direction in ["left", "right"] * 4
    ]


def build_scan_trials(run_number):
    """28 trials: 8 FIPS, 4 Veridical, 8 Perceptual, and 8 FrameOnly.

    Three randomized mini-blocks contain 3, 3, and 2 FIPS trials,
    respectively, plus 3/3/2 Perceptual, 2/1/1 Veridical, and 3/3/2 FrameOnly.
    """
    fips = np.random.permutation(eight_fips_trials()).tolist()
    perceptual = np.random.permutation(eight_perceptual_trials()).tolist()
    veridical = four_veridical_trials()
    frame_only = np.random.permutation(eight_frame_only_trials()).tolist()

    all_trials = []

    fips_start = 0
    veridical_start = 0
    for block, fips_count in enumerate((3, 3, 2), start=1):
        veridical_count = (2, 1, 1)[block - 1]
        block_trials = (
            fips[fips_start:fips_start + fips_count]
            + veridical[veridical_start:veridical_start + veridical_count]
            + perceptual[fips_start:fips_start + fips_count]
            + frame_only[fips_start:fips_start + fips_count]
        )

        veridical_start += veridical_count
        fips_start += fips_count
        block_trials = np.random.permutation(block_trials).tolist()

        for t in block_trials:
            t["Block"] = block

        all_trials.extend(copy.deepcopy(block_trials))

    for idx, trial in enumerate(all_trials, start=1):
        trial["Run"] = run_number
        trial["Trial"] = idx

    return all_trials


def build_prescan_trials():
    factor_prescan = {
        "ExpCondition": [cfg.COND_FIPS],
        "InitialPosition": ["upper", "lower"],
        "InducedPercept": ["left", "right"],
    }

    cond_list = data.createFactorialTrialList(factor_prescan)
    n_reps = max(round(exp["n_trials"] // len(cond_list)), 1)
    trials = cond_list * n_reps
    trials = np.random.permutation(trials).tolist()
    trials = trials[: exp["n_trials"]]

    for idx, trial in enumerate(trials, start=1):
        trial["Block"] = 1
        trial["Run"] = int(expInfo["run"])
        trial["Trial"] = idx
        trial["InitialDirection"] = source_initial_direction(
            trial["InitialPosition"],
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

pd.DataFrame(TRIALS).to_csv(trialsName, index=False)


# ===========================================================================
# ONE-HEMIFIELD STIMULUS CONFIGURATION
# ===========================================================================

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
    # pRF uses integer pixel offsets from display center.
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


# ===========================================================================
# GENERATE FIXATION
# ===========================================================================

# Exact Kay runretinotopy fixation dimensions:
# fixationsize = 4 pixels, and ptviewmovie defines scalar fixationsize
# as the fixation-dot diameter with no border.
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

# Invisible placeholder retained only so the surrounding fips_stim.py-style
# infrastructure can keep using fix_inner.autoDraw without a broad refactor.
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


# ===========================================================================
# GENERATE DOTS / MARKERS
# ===========================================================================

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

            # PsychoPy 2026.2.4 compatibility:
            # visual.Pie still forwards legacy color parameters internally.
            # Supplying the undefined sentinel prevents those deprecated
            # parameters from overriding the explicit colorSpace below.
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


# ===========================================================================
# FRAME
# ===========================================================================

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


# ===========================================================================
# INSTRUCTIONS
# ===========================================================================

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


# ===========================================================================
# STIMULUS HELPERS
# ===========================================================================

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
        dot.fillColor = color
        dot.lineColor = color


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
    dot_pos,
    dot_counter=0,
):
    n_one_way, time_left = divmod(
        time_elapsed,
        stim_config["duration_one_way"],
    )

    current_color = dot_color
    current_pos = dot_pos

    if time_left > exp["reversal_time"]:
        if int(n_one_way) % 2 == 1:
            if expInfo["stimulus"] == "old":
                current_color = (
                    "black" if dot_color == "white" else "white"
                )

            current_pos = (
                "upper" if dot_pos == "lower" else "lower"
            )

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
    """
    Same opposite-horizontal-offset convention as the source experiment.
    """
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
    """
    Clock-based triangular frame movement copied from the source logic.
    """
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


# ===========================================================================
# EYELINK / SCREEN HELPERS
# ===========================================================================

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
    """
    Gracefully stop tracking, preserve fixation-task data, close the EDF,
    transfer it, then quit.
    """
    # Preserve attention-task data even on an aborted run.
    # The globals check keeps this safe if termination happens before
    # the scan fixation task has been initialized.
    if (
        "fixation_task" in globals()
        and fixation_task is not None
        and "close_fixation_task_log" in globals()
    ):
        try:
            close_fixation_task_log()
        except Exception as err:
            print(
                "WARNING: could not finalize fixation-task log:",
                err,
            )

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


# ===========================================================================
# KAY RETINOTOPY FIXATION-COLOR ATTENTION TASK
# ===========================================================================

fixation_task = None
fixation_task_csv_handle = None
fixation_task_csv_writer = None

FIXATION_TASK_FIELDS = [
    "event_type",
    "run_time_s",
    "scheduled_time_s",
    "color",
    "key",
]


def _matlab_round_positive(x):
    """Equivalent to MATLAB round() for the positive values used here."""
    return int(np.floor(float(x) + 0.5))


def _sample_kay_fixation_interval_s():
    """
    Reproduce the source rule:

        soafun = @() round(
            meanchange*(60/frameduration)
            + changeplusminus*(2*(rand-.5))*(60/frameduration)
        )

    with frameduration=4, meanchange=3, changeplusminus=2.
    """
    hz = cfg.FIXATION_TASK_HZ

    interval_frames = _matlab_round_positive(
        cfg.FIXATION_CHANGE_MEAN_S * hz
        + cfg.FIXATION_CHANGE_PLUS_MINUS_S
        * (2.0 * (np.random.rand() - 0.5))
        * hz
    )

    return max(interval_frames, 1) / hz


def _set_fixation_color(index):
    color = cfg.FIXATION_COLORS_RGB255[index]
    fix_outer.fillColor = color
    fix_outer.lineColor = color


def _open_fixation_task_log():
    """
    Create the fixation-task CSV immediately at run start.

    The file stays open during the run. Each event is appended and flushed
    immediately, so stopping the run midway does not discard earlier
    attention-task events.
    """
    global fixation_task_csv_handle
    global fixation_task_csv_writer

    # Close a stale handle first, if one somehow exists.
    if fixation_task_csv_handle is not None:
        try:
            fixation_task_csv_handle.flush()
            fixation_task_csv_handle.close()
        except Exception:
            pass

    fixation_task_csv_handle = open(
        fixationTaskName,
        "w",
        newline="",
        encoding="utf-8",
    )

    fixation_task_csv_writer = csv.DictWriter(
        fixation_task_csv_handle,
        fieldnames=FIXATION_TASK_FIELDS,
    )
    fixation_task_csv_writer.writeheader()
    fixation_task_csv_handle.flush()


def _record_fixation_event(event):
    """
    Store one fixation-task event in memory AND append it to disk immediately.
    """
    if fixation_task is None:
        return

    fixation_task["event_log"].append(event)

    if fixation_task_csv_writer is not None:
        fixation_task_csv_writer.writerow(event)
        fixation_task_csv_handle.flush()


def start_fixation_task(start_run_s=0.0):
    """
    Kay's implementation starts with the first fixation color: red.
    """
    global fixation_task

    _set_fixation_color(0)

    fixation_task = {
        "current_color_index": 0,
        "next_change_run_s": start_run_s + _sample_kay_fixation_interval_s(),
        "change_count": 0,
        "event_log": [],
    }

    # Create the CSV right now, before the first baseline begins.
    _open_fixation_task_log()

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
    """
    When a change is due, pick randomly from the OTHER two colors.
    This preserves the source behavior: every event is a real color change.
    """
    if fixation_task is None:
        return

    while run_time_s >= fixation_task["next_change_run_s"]:
        current = fixation_task["current_color_index"]
        available = [
            idx
            for idx in range(len(cfg.FIXATION_COLORS_RGB255))
            if idx != current
        ]

        new_index = int(np.random.permutation(available)[0])
        scheduled = fixation_task["next_change_run_s"]

        fixation_task["current_color_index"] = new_index
        fixation_task["change_count"] += 1
        fixation_task["next_change_run_s"] = (
            scheduled + _sample_kay_fixation_interval_s()
        )

        _set_fixation_color(new_index)

        _record_fixation_event(
            {
                "event_type": "color_change",
                "run_time_s": float(run_time_s),
                "scheduled_time_s": float(scheduled),
                "color": cfg.FIXATION_COLOR_NAMES[new_index],
                "key": "",
            }
        )

        el_tracker.sendMessage(
            f"FIXATION_COLOR_CHANGE "
            f"{fixation_task['change_count']} "
            f"{cfg.FIXATION_COLOR_NAMES[new_index]}"
        )


def collect_fixation_task_keys():
    """
    Log button presses. The original public runretinotopy instructions do not
    specify one particular response key, only that a button be pressed when
    the fixation color changes.
    """
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


def save_fixation_task_log():
    """
    Flush the already-incremental attention log to disk.

    Kept as a named function because the rest of the experiment already calls
    it at the normal end of a run.
    """
    if fixation_task_csv_handle is not None:
        fixation_task_csv_handle.flush()
        return

    # Fallback only: if there is task data but no open writer, materialize it.
    if fixation_task is not None:
        pd.DataFrame(
            fixation_task["event_log"],
            columns=FIXATION_TASK_FIELDS,
        ).to_csv(
            fixationTaskName,
            index=False,
        )


def close_fixation_task_log():
    """
    Flush and close the fixation-task CSV safely.
    """
    global fixation_task_csv_handle
    global fixation_task_csv_writer

    if fixation_task_csv_handle is not None:
        fixation_task_csv_handle.flush()
        fixation_task_csv_handle.close()

    fixation_task_csv_handle = None
    fixation_task_csv_writer = None


def prepare_fixation_only_screen():
    """
    Force a true fixation-only display.

    Baselines and ITIs must contain:
        - fixation dot
        - background

    They must NOT contain:
        - moving/static frame
        - FIPS probes
        - perceptual/veridical probes
        - prescan markers
    """
    hide_all_dots(dots)
    hide_all_dots(markers)

    frame_outer.autoDraw = False
    frame_inner.autoDraw = False

    fix_outer.autoDraw = True
    fix_inner.autoDraw = True


def run_fixation_until(target_run_s):
    """
    Keep the fixation-color task running during pre-run baseline, ITIs,
    and post-run baseline, while guaranteeing that only fixation is visible.
    """
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
    """
    Optional EyeLink fixation gate retained from the source experiment.
    """
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

        # Preserve the source rectangular gate.
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


# ===========================================================================
# SCAN TRIAL
# ===========================================================================

def show_stimulus(
    trial,
    when_to_flip,
    add_iti_after=True,
):
    """
    Run one 8-s scanner trial.

    The only condition-specific changes are stimulus visibility/position:
        FIPS       -> moving frame + aligned probes
        Veridical  -> no frame + aligned probes
        Perceptual -> no frame + physically displaced probes
        FrameOnly  -> moving frame + no probes

    EyeLink, timing, responses, logging, and the scanner schedule stay shared.
    """

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

    # Keep the Kay fixation task active during the pre-run baseline / ITI.
    # run_fixation_until() intentionally disables the frame and probes so the
    # baseline/ITI is fixation-only.
    if expInfo["ses"] == "scan":
        run_fixation_until(when_to_flip)
    else:
        wait_duration = when_to_flip - exp_clock.getTime()
        if wait_duration > 0:
            core.wait(wait_duration)

    # IMPORTANT: the fixation-only wait above turns the frame off. Restore
    # the condition-specific frame state immediately before the stimulus
    # begins. Without this, FIPS and FrameOnly trials have frame positions
    # updated in memory but the frame is never actually drawn.
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

    expWin.recordFrameIntervals = True
    trial_clock = core.Clock()
    trial_clock.reset()

    dot_counter = 0
    onset_sent = False


    while trial_clock.getTime() < stim_config["duration_stimulus"]:
        time_elapsed = trial_clock.getTime()

        # Kay fixation-color task runs continuously during stimulus presentation.
        update_fixation_task(exp_clock.getTime())

        # Probe update.
        if probes_present:
            _, dot_counter = update_dots_by_clock(
                time_elapsed,
                dot_color,
                dot_pos,
                dot_counter,
            )
        else:
            hide_all_dots()

        # Frame update.
        if frame_present:
            shift_frames_by_clock(
                dir_factor,
                time_elapsed,
                frame_default_pos,
            )

        # Mark first visual frame in EDF.
        if not onset_sent:
            expWin.callOnFlip(
                el_tracker.sendMessage,
                f"STIM_ON {nt_label}",
            )

        expWin.flip()

        if not onset_sent:
            trial["stim_onset_run_s"] = exp_clock.getTime()
            onset_sent = True

        # Kay fixation-color attention response.
        if collect_fixation_task_keys():
            terminate_task()

    event.clearEvents()

    # End every trial on a guaranteed fixation-only screen.
    prepare_fixation_only_screen()

    expWin.recordFrameIntervals = False

    # This flip begins the fixation-only ITI.
    expWin.flip()

    trial["stim_offset_run_s"] = exp_clock.getTime()


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


# ===========================================================================
# PRESCAN MATCHING
# ===========================================================================

def test_stimulus(trial):
    """
    Prescan psychophysical matching, retained from the source design.

    The display alternates between:
        FIPS stimulus
        adjustable real-displacement comparison

    The participant adjusts the comparison separation and accepts it.
    """
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
                    dot_pos,
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


# ===========================================================================
# RUN
# ===========================================================================

if expInfo["ses"] == "scan":
    if use_fake_prescan:
        show_msg(expWin, prescan_text, True)

    elif not predataName.exists():
        show_msg(
            expWin,
            "WARNING: no prescan summary was found.\n\n"
            "The Perceptual condition cannot run correctly "
            "without participant-specific displacement values.\n\n"
            "Real scan mode requires a participant prescan.",
            True,
        )

    else:
        show_msg(expWin, prescan_text, True)

    show_msg(expWin, task_text, True)


# ---------------------------------------------------------------------------
# EyeLink calibration
# ---------------------------------------------------------------------------

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

else:
    eyelink_msg = (
        "Running EyeLink in Dummy mode.\n\n"
        "Press any key to continue."
    )
    show_msg(expWin, eyelink_msg)


# ---------------------------------------------------------------------------
# Scanner trigger
# ---------------------------------------------------------------------------

show_msg(
    expWin,
    (
        f"Waiting for scanner signal "
        f"({cfg.SCANNER_TRIGGER_KEY})..."
    ),
    False,
)

event.waitKeys(keyList=[cfg.SCANNER_TRIGGER_KEY])

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

# The run begins with a true fixation-only baseline.
prepare_fixation_only_screen()

expWin.color = stim_config["backColor"]
expWin.flip(clearBuffer=True)

# Record eye position for the entire run, including baseline.
el_tracker.startRecording(1, 1, 1, 1)
el_tracker.sendMessage("RUN_START")

if expInfo["ses"] == "scan":
    start_fixation_task(start_run_s=0.0)

# First stimulus onset is 15 s after scanner trigger in the scan session.
when_to_flip = exp["pre_run_baseline"]

logFile.write(
    f"Pre-run fixation baseline = "
    f"{exp['pre_run_baseline']:.3f} s\n"
)


# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------

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
    pd.DataFrame(dataFile).to_csv(
        dataName,
        index=False,
    )


# ---------------------------------------------------------------------------
# Post-run baseline
# ---------------------------------------------------------------------------

if expInfo["ses"] == "scan":
    post_baseline_end = (
        when_to_flip + exp["post_run_baseline"]
    )

    run_fixation_until(post_baseline_end)
    close_fixation_task_log()

    el_tracker.sendMessage("RUN_END")

    logFile.write(
        f"Post-run fixation baseline = "
        f"{exp['post_run_baseline']:.3f} s\n"
    )


# ===========================================================================
# WRAP UP
# ===========================================================================

t_end = exp_clock.getTime()
exp_duration = t_end - t_start

logFile.write(
    f"The experiment took {exp_duration:.3f} seconds\n"
)

expWin.saveFrameIntervals(
    fileName=str(framesName)
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

    prescan_summary = pd.DataFrame(
        [
            {
                "left": left_mean,
                "right": right_mean,
                "avg": avg_abs,
            }
        ]
    )

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


# Stop recording, transfer EDF, and quit.
if el_tracker.isRecording():
    pylink.pumpDelay(100)
    el_tracker.stopRecording()

terminate_task()

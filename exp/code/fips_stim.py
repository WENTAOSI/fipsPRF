#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
FIPS PRF with one pair of dots

Lastly updated: Nov 20 2025
@author: Eunhye Choe

modifications:
Apr 15 2025: Skip fixation check
Apr 17 2025: Add Checkerboard dot option
June 4 2025: Modify for fMRI scan
June 11 2025: Update prescan session
Nov 17 2025: Update the main task
Nov 21 2025: Randomized the color of checkerboard

@author: Wentao Si 

modifications: accomodate the design of one pair of dots to maximize effect 
"""

# %% Debugging? 
# ---------------------------------------------------------------------------
CODE_TEST = False; DEBUG_ON_VSCODE = False
TASK_TILT = True

# Preset for debugging
# ---------------------------------------------------------------------------
if DEBUG_ON_VSCODE: from update_sys_path import update_system_path; update_system_path()
if TASK_TILT: tilt = True

# ===========================================================================
#
#                               SYSTEM SETUP                                 
#
# ===========================================================================

# %% IMPORT LIBRARIES
# ---------------------------------------------------------------------------

# Import packages
from psychopy import visual, event, core, monitors, logging, gui, data
import numpy as np
import copy
import os
import pandas as pd
from pathlib import Path

# Required packages for eyelink
import pylink
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


# %% GLOBAL VARIABLES
# ---------------------------------------------------------------------------

# Session Info
TASK = "FIPS" 
expInfo = {'sub_id': '99','sub_name': 'PT', 'run': '1'} # default values

# Auto-update
#ROOTDIR = Path('.').resolve()  # find the current directory
ROOTDIR = Path(__file__).resolve().parent # find the current file dir
DATADIR = ROOTDIR.parent / "data" / "psycphys"

# find the latest sub id
sub_dirs = sorted([d for d in DATADIR.iterdir() if d.is_dir() and d.name.startswith("sub-")],
                  key=os.path.getctime, reverse=True)

if sub_dirs:
    recent_subdir = sub_dirs[0]
    sub_id = recent_subdir.name.replace("sub-", "")

    # find the latest run
    csv_files = sorted(recent_subdir.glob("*.csv"), key=os.path.getctime, reverse=True)
    if csv_files:
        recent_file = csv_files[0]
        name_parts = recent_file.stem.split("_") # e.g., sub-1_PT_FIPS_prescan_1_data.csv
        if len(name_parts) >= 5:
            file_sub_name = name_parts[1]
            file_session = name_parts[3]
            file_run = name_parts[4]

            if file_run.isdigit() and file_run != "10":
                expInfo["sub_id"] = sub_id.zfill(2)
                expInfo["sub_name"] = file_sub_name
                if file_session == "prescan":
                    expInfo["run"] = file_run  # keep as is
                elif file_session == "scan":
                    expInfo["run"] = str(int(file_run) + 1)  # increment             


# Monitor Info
my_monitors = {
    "Macbook": {"mon_dist": 50, "size_cm": (42, 0), "size_px": (1440, 900), "refresh_rate": 60, "screen": 1},
    "iMac": {"mon_dist": 60, "size_cm": (30, 0), "size_px": (1920, 1080), "refresh_rate": 60, "screen": 1},
    "TseLab": {"mon_dist": 63, "size_cm": (47.2, 0), "size_px": (1920, 1080), "refresh_rate": 60, "screen": 1},
    "Beaver": {"mon_dist": 80, "size_cm": (52.3, 0), "size_px": (1920, 1080), "refresh_rate": 60, "screen": 1},
    "Scanner": {"mon_dist": 128.7, "size_cm": (42.8, 0), "size_px": (1920, 1080), "refresh_rate": 60, "screen": 1}
}

# Create GUI
dlg = gui.Dlg(title=TASK)
dlg.addField("Subject ID:", expInfo['sub_id'])
dlg.addField("Initials:", expInfo['sub_name'])
dlg.addField("Session:", choices=["prescan", "scan"], initial="scan")
dlg.addField("Monitor:", choices=list(my_monitors.keys()), initial="Scanner")
dlg.addField("Stimulus:", choices=["plain", "checkerboard", "old"], initial="checkerboard")
dlg.addField("Eyelink Connected:", choices=["no", "yes"], initial="yes")
dlg.addField("Run:", expInfo['run'])
dlg.show()  # Show GUI
if dlg.OK == False: core.quit()  # user pressed cancel

# Update expInfo with selected values
expInfo.update({
    'exp_name': TASK,
    'sub_id': dlg.data[0],
    'sub_name': dlg.data[1],
    'ses': dlg.data[2],
    'monitor': dlg.data[3],
    'stimulus': dlg.data[4],
    'eyelink': dlg.data[5],
    'run': dlg.data[6],
    'date': data.getDateStr()
})

# Set the parameters
FLASH_GRAB = True if expInfo['stimulus'] == "checkerboard" else False
EYELINK_CONNECTED = True if expInfo['eyelink'] == "yes" else False

# %% SAVING and LOGGING
# ---------------------------------------------------------------------------

# Directories and files in BIDS format
sub_id = f"{int(expInfo['sub_id']):02d}"
run_id = f"{int(expInfo['run']):02d}"
#ROOTDIR = Path('.').resolve()  # find the current directory
ROOTDIR = Path(__file__).resolve().parent # find the current file dir
DATADIR = ROOTDIR.parent / "data" / "psycphys"
if not DATADIR.exists():
    DATADIR.mkdir(parents=True, exist_ok=True)
    
SUBDIR = DATADIR / F"sub-{sub_id}"
LOGDIR = DATADIR / "logs"
EDFDIR = DATADIR / "edf"
# create directories if not exist

if not SUBDIR.exists():
    SUBDIR.mkdir(parents=True, exist_ok=True)
if not LOGDIR.exists():
    LOGDIR.mkdir(parents=True, exist_ok=True)
if not EDFDIR.exists():
    EDFDIR.mkdir(parents=True, exist_ok=True)



# Create directories if not exsit
for directory in [DATADIR, SUBDIR, LOGDIR, EDFDIR]:
    os.makedirs(directory, exist_ok=True)

# Save the design matrix and data in csv format
trialsName = SUBDIR / f"sub-{sub_id}_{expInfo['sub_name']}_{expInfo['exp_name']}_{expInfo['ses']}_{expInfo['run']}_psychophysics.csv"
dataName = SUBDIR / f"sub-{sub_id}_{expInfo['sub_name']}_{expInfo['exp_name']}_{expInfo['ses']}_{expInfo['run']}_data.csv"

# Save a log file
framesName = LOGDIR / f"sub-{sub_id}_{expInfo['exp_name']}_{expInfo['ses']}_{expInfo['run']}.csv"
logName = LOGDIR / f"sub-{sub_id}_{expInfo['exp_name']}_{expInfo['ses']}_{expInfo['run']}.log"

# Start Logging
logFile = logging.LogFile(str(logName), level=logging.EXP)
logFile.write(f"ID: {expInfo['sub_id']},Subject: {expInfo['sub_name']}, Experiment: {expInfo['exp_name']}, Session: {expInfo['ses']}"+ '\n')
logging.console.setLevel(logging.WARNING)  # Set console logging level

# Eyelink file name
edf_fname = f"{sub_id}{TASK}{run_id}"
edf_file = edf_fname + ".EDF" 

# Eyelink session directory
session_identifier = f"sub-{sub_id}_{expInfo['sub_name']}_{expInfo['exp_name']}_{expInfo['ses']}"
session_folder = os.path.join(EDFDIR, session_identifier)
local_edf = os.path.join(session_folder, session_identifier + '.EDF')
os.makedirs(session_folder, exist_ok=True)

# load prescan summary data
PRESCAN_LEFT = 0
PRESCAN_RIGHT = 0
PRESCAN_AVG = 0
predataName = SUBDIR / f"sub-{sub_id}_{expInfo['sub_name']}_{expInfo['exp_name']}_prescan_summary.csv"
if predataName.exists():
    prescan_summary_df = pd.read_csv(predataName)
    PRESCAN_LEFT = prescan_summary_df.loc[0, "left"]
    PRESCAN_RIGHT = prescan_summary_df.loc[0, "right"]
    PRESCAN_AVG = prescan_summary_df.loc[0, "avg"]
    prescan_text = f"Loaded prescan summary:\n Left: {PRESCAN_LEFT:.2f}, Right: {PRESCAN_RIGHT:.2f}, Avg: {PRESCAN_AVG:.2f}"
else:
    prescan_text = f"Prescan summary file not found: {predataName}"

# %% MONITOR and WINDOW
# ---------------------------------------------------------------------------

# Set Monitor Info
monName = expInfo['monitor']
monInfo = my_monitors[monName]

# Initialize Monitor
expMon = monitors.Monitor(name=monName,
                           width=monInfo["size_cm"][0],
                           distance=monInfo["mon_dist"])
expMon.setSizePix(monInfo["size_px"])
expMon.save()

# Set Screen
expWin = visual.Window(monitor=expMon,
                        size=monInfo["size_px"],
                        screen=monInfo["screen"],
                        fullscr=False,
                        units='pix', # pixel unit for Eyelink
                        winType='pyglet',
                        allowGUI=False,
                        allowStencil=False,
                        color=[-1,-1,-1],
                        colorSpace='rgb',
                        blendMode='avg',
                        waitBlanking=True
                        )

# Log Monitor Info
logFile.write(f"Monitor: {monName}\n")
logFile.write(f"Distance = {monInfo['mon_dist']} cm\n, Width = {monInfo['size_cm'][0]} cm\n")
logFile.write(f"Pixel Width = {monInfo['size_px'][0]}, Pixel Height = {monInfo['size_px'][1]}\n")
logFile.write(f"Refresh Rate = {monInfo['refresh_rate']} Hz\n")

# Calculate pixel per degree (for compatibility with Eyelink)
screen_width_cm = monInfo["size_cm"][0]  # Screen width in cm
screen_width_px = monInfo["size_px"][0]  # Screen width in pixels
viewing_distance_cm = monInfo["mon_dist"]  # Viewing distance in cm

# formula: cm = 2 * d * tan(dva / 2)
px_per_cm = screen_width_px / screen_width_cm
cm_per_dva = 2 * viewing_distance_cm * np.tan(np.radians(1) / 2)
ppd = cm_per_dva * px_per_cm

# Convert degrees to pixels
def deg2pix(dva):
    if isinstance(dva, (list, tuple, np.ndarray)):
        return [x * ppd for x in dva]
    else:
        return dva * ppd

# ===========================================================================

#                              EYELINK SETUP

# ===========================================================================

# %% Step 1: Connect to the EyeLink Host PC
# ---------------------------------------------------------------------------
if EYELINK_CONNECTED:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()
else:
    el_tracker = pylink.EyeLink(None)


# %% Step 2: Open an EDF data file on the Host PC
# ---------------------------------------------------------------------------
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file
experiment_info = (
    f"RECORDED BY {os.path.basename(__file__)}\n"
    f"Subject ID: {expInfo['sub_id']}\n"
    f"Session: {expInfo['ses']}\n"
    f"Monitor: {expInfo['monitor']}\n"
    f"Date: {expInfo['date']}"
)
el_tracker.sendCommand(f"add_file_preamble_text '{experiment_info}'")


# %% Step 3: Configure the tracker
# ---------------------------------------------------------------------------

# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000, 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if EYELINK_CONNECTED:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# ---------------------------------------------------------------------------
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# ---------------------------------------------------------------------------
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
# if eyelink_ver > 2:
#     el_tracker.sendCommand("sample_rate 1000")
# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV9")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

# %% Step 4: set up a graphics environment for calibration
# ---------------------------------------------------------------------------

# get the native screen resolution used by PsychoPy
scn_width, scn_height = expWin.size

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, expWin)
print(genv)  # print out the version number of the CoreGraphics library

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = (0.2, 0.2, 0.2)
genv.setCalibrationColors(foreground_color, background_color)

# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
#
# Use gen.setPictureTarget() to set a "picture" target
# genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
#
# Use genv.setMovieTarget() to set a "movie" target
# genv.setMovieTarget(os.path.join('videos', 'calibVid.mov'))

# Use the default calibration target ('circle')
genv.setTargetType('circle')

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle", "spiral", and "movie" targets
genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

# Backdrop for the host display
left = int(scn_width/2.0) - 60
top = int(scn_height/2.0) - 60
right = int(scn_width/2.0) + 60
bottom = int(scn_height/2.0) + 60
draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)


# %% define a few helper functions for trial handling
# ---------------------------------------------------------------------------

def clear_screen(win):
    """ clear up the PsychoPy window"""
    event.clearEvents()
    win.fillColor = genv.getBackgroundColor()
    for marker in markers: draw_dot(marker, False)
    for dot in dots: draw_dot(dot, False)
    fix_outer.autoDraw = False; fix_inner.autoDraw = False
    frame_outer.autoDraw = False; frame_inner.autoDraw = False
    win.flip()

def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""

    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(expWin, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    expWin.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(expWin)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR

# ===========================================================================

#                          EXPERIMENT PROCEDURE

# ===========================================================================

# %% EXPERIMENT PARAMETERS
# ---------------------------------------------------------------------------

PARAMETER_PRESCAN = {
    # Trial
    "n_trials": 16,
    "n_runs": 1,
    "n_blocks": 1,

    # Stimuli
    "cycle": 8,

    # Timing
    "dummy_time": 0,
    "dot_duration": 50 / 1000, # ms
    "reversal_time": 450 / 1000,
 #   "response_delay": 150 / 1000,
 #   "report_time": 5, # s
 #   "feedback_time": 1,
    "ITI": 2/ 1000, 
}


PARAMETER_SCAN = {
    # Trial
    "n_trials": 30,
    "n_runs": 10,
    "n_blocks": 1,

    # Stimuli
    "cycle": 8,

    # Timing
    "dummy_time": 8,
    "dot_duration": 50 / 1000, # ms
    "reversal_time": 450 / 1000,
 #   "response_delay": 150 / 1000,
 #   "report_time": 5, # s
 #   "feedback_time": 1,
    "ITI": 8,
}

# %% COUNTERBALANCING FACTORS
# ---------------------------------------------------------------------------

FACTOR_PRESCAN = {
    "ExpCondition": ["Illusion"],
    "InitialPosition": ["upper", "lower"],
    "InducedPercept": ["left", "right"]    
    }

FACTOR_ILLUSION = {
    "ExpCondition": ["Illusion"],
    "InitialPosition": ["upper", "lower"],
    "InducedPercept": ["left", "right"]
}

FACTOR_PHYSICAL = {
    "ExpCondition": ["Physical"],
    "InitialPosition": ["upper", "lower"],
    "InducedPercept": ["none"]
}

FACTOR_PERCEPTUAL = {
    "ExpCondition": ["Perceptual"],
    "InitialPosition": ["upper", "lower"],
    "InducedPercept": ["left", "right"]
}

COND_ILLUSION = data.createFactorialTrialList(FACTOR_ILLUSION)
COND_PHYSICAL = data.createFactorialTrialList(FACTOR_PHYSICAL)
COND_PERCEPTUAL = data.createFactorialTrialList(FACTOR_PERCEPTUAL)

# --------- factor list -------
# InitialPosition,InitialDirection
# ------------------------------
# upper,left (left)
# lower,left (right)
# upper,right (right)
# lower,right (left)
# ------------------------------

# %% GENERATE TRIAL LIST
# ---------------------------------------------------------------------------

# Setup parameters and factors
exp = PARAMETER_SCAN if expInfo['ses'] == "scan" else PARAMETER_PRESCAN
if expInfo['ses'] == "scan":
    exp = PARAMETER_SCAN
    COND_LIST = COND_ILLUSION + COND_PHYSICAL + COND_PERCEPTUAL
else:
    exp = PARAMETER_PRESCAN
    COND_LIST = data.createFactorialTrialList(FACTOR_PRESCAN)

# Reset for debugging
if CODE_TEST:
    exp["n_trials"] = 4

# Generate a list of conditions for trials
TRIALS = []
for block in range(1, exp["n_blocks"]+1):
    n_reps = max(round(exp["n_trials"]//len(COND_LIST)), 1)
    trials = COND_LIST * n_reps

    # Label blocks
    for trial in trials:
        trial["Block"] = block

    # Shuffle
    shuffled_trials = np.random.permutation(trials).tolist()

    # Truncate if needed
    if len(shuffled_trials) > exp["n_trials"]:
        shuffled_trials = shuffled_trials[:exp["n_trials"]]

    TRIALS.extend([copy.deepcopy(trial) for trial in shuffled_trials])

# trial number and condition labeling
for idx, trial in enumerate(TRIALS, start=1):
    trial["Run"] = int(expInfo['run'])
    trial["Trial"] = idx
    if trial["ExpCondition"] == "Illusion":
        if (trial["InitialPosition"] == "upper" and trial["InducedPercept"] == "left") or \
            (trial["InitialPosition"] == "lower" and trial["InducedPercept"] == "right"):
            trial["InitialDirection"] = "right"
        else:
            trial["InitialDirection"] = "left"
    else:
        trial["InitialDirection"] = "NA"


# Save trial list
pd.DataFrame(TRIALS).to_csv(trialsName, index=False)


# ===========================================================================

#                             STIMULUS SETUP                          

# ===========================================================================

# %% STIMULI PARAMETERS
# ---------------------------------------------------------------------------

# ============================================================

# SPATIAL LAYOUT

# ============================================================

FIX_POS_DVA  = [-6.0, 0.0]   # fixation on far left
STIM_POS_DVA = [ 2.0, 0.0]   # FIPS stimulus on right

config = {
    # General
    "backColor": [0.2, 0.2, 0.2],
    "session": expInfo["ses"],
    # Fixation -- completely independent
    "fix_pos": deg2pix(FIX_POS_DVA),
    "fix_color": [0.3, 0.3, 0.3],
    "fix_innerRad": deg2pix(0.05),
    "fix_outerRad": deg2pix(0.125),
    # FIPS stimulus center
    "stim_pos": deg2pix(STIM_POS_DVA),
    # Probe/marker center
    "mark_pos": deg2pix(STIM_POS_DVA),
    # Dots
    "dot_radius": deg2pix(0.5),
    "dot_diameter": deg2pix(1.0),
    "center_offset_y": deg2pix(2.0),
    # Frame
    "frame_width": deg2pix(8.0),
    "frame_height": deg2pix(8.0),
    "frame_linewidth": deg2pix(0.25),
    "frame_color": [0.3, 0.3, 0.3],
    # Total horizontal frame excursion
    "frame_travel": deg2pix(7.0),
    # Timing
    "frame_per_sec": monInfo["refresh_rate"],
    "sec_per_frame": 1 / monInfo["refresh_rate"],
    "n_frames_per_move": exp["reversal_time"] * monInfo["refresh_rate"],
    "n_frames_per_pause": exp["dot_duration"] * monInfo["refresh_rate"],
}

# Fixation contrast color
delta = 0.20
config["fix_dec"] = np.clip(np.array(config['fix_color']) * (1 - delta), 0, 1)
config["fix_inc"] = np.clip(np.array(config['fix_color']) * (1 + delta), 0, 1)


# Calculate the frame motion
config["frame_travel"] = deg2pix(7.0)
'''
(config["frame_width"]
    - 2 * config["center_offset_x"]
    - 2 * config["frame_offset_x"])
'''
config["shift_pix_per_frame"] = config["frame_travel"] / config["n_frames_per_move"]
config["shift_pix_per_second"] = config["shift_pix_per_frame"] * config["frame_per_sec"]

config["duration_one_way"] = exp["reversal_time"] + exp["dot_duration"]
config["duration_stimulus"] = config["duration_one_way"] * 2 * exp["cycle"]
config["duration_one_block"] = config["duration_stimulus"] + exp["ITI"]

# Refresh the window setting (because Eyelink overrides it)
expWin.colorSpace = 'rgb'
expWin.color = config["backColor"]
expWin.refreshThreshold = 1/monInfo['refresh_rate'] + 0.003
expWin.units = 'pix'

# %% GENERATE FIXATION
# ---------------------------------------------------------------------------

fix_inner = visual.Circle(win=expWin, radius=config["fix_innerRad"], pos=config["fix_pos"],
                          fillColor=config["backColor"], lineColor=config["backColor"], autoLog=False)
fix_outer = visual.Circle(win=expWin, radius=config["fix_outerRad"], pos=config["fix_pos"],
                          fillColor=config["fix_color"], lineColor=config["fix_color"], autoLog=False)


# %% GENERATE DOTS
# ---------------------------------------------------------------------------

dot_positions = [
    [config["stim_pos"][0], config["stim_pos"][1] + config["center_offset_y"]],  # upper
    [config["stim_pos"][0], config["stim_pos"][1] - config["center_offset_y"]]   # lower
]

dots = []
for pos in dot_positions:
    dot = visual.Circle(win=expWin, radius=config["dot_radius"], pos=pos,
                        fillColor="white", lineColor="white", autoLog=False)
    dots.append(dot)

# %% DOTS AT PERCEIVED LOCATION
# ---------------------------------------------------------------------------

dots = []
for pos in dot_positions:
    dot = visual.Circle(win=expWin, radius=config["dot_radius"], pos=pos,
                        fillColor="white", lineColor="white", autoLog=False)
    dots.append(dot)

# %% GENERATE MARKER DOTS
# ---------------------------------------------------------------------------
marker_positions = [
    [config["mark_pos"][0], config["mark_pos"][1] + config["center_offset_y"]],  # upper
    [config["mark_pos"][0], config["mark_pos"][1] - config["center_offset_y"]]   # lower
]

markers = []
for pos in marker_positions:
    marker = visual.Circle(win=expWin, radius=config["dot_radius"], pos=pos,
                        fillColor="white", lineColor="white", autoLog=False)
    markers.append(marker)

mouseResp = event.Mouse()
mouseResp.autoLog = False

# %% ADD CHECKERBOARD PATTERN (if needed)
# ---------------------------------------------------------------------------
if FLASH_GRAB:
    def make_checker_dot(win, pos, radius, rot=False):
        """
        Creates a 4-quadrant black-and-white checker dot at given position.
        """
        colors = [["black", 0, 90], ["white", 90, 180],
                ["black", 180, 270], ["white", 270, 360]]
        
        pies = []
        for col, ang1, ang2 in colors:
            pie = visual.Pie(win, radius=radius, start=ang1, end=ang2,
                            fillColor=col, lineColor=col, pos=pos, autoLog=False)
            pie.base_start = ang1
            pie.base_end = ang2
            pies.append(pie)
        return pies

    dots = []
    for pos in dot_positions:
        dot_quadrants = make_checker_dot(expWin, pos, config["dot_radius"])
        dots.append(dot_quadrants) 

    markers = []
    for pos in marker_positions:
        marker = make_checker_dot(expWin, pos, config["dot_radius"])
        markers.append(marker) 

def rotate_checker(dot_quadrants, angle_deg):
    for pie in dot_quadrants:
        pie.start = pie.base_start + angle_deg
        pie.end   = pie.base_end + angle_deg

# %% GENERATE FRAME
# ---------------------------------------------------------------------------

# generate frame
frame_outer = visual.Rect(win=expWin,
                    width=config["frame_width"]+config["frame_linewidth"],
                    height=config["frame_height"]+config["frame_linewidth"],
                    lineColor=config["frame_color"], lineWidth=1, interpolate = False,
                    fillColor=config["frame_color"], autoLog=False, depth=2)
frame_inner = visual.Rect(win=expWin,
                          width=config["frame_width"], height=config["frame_height"],
                            lineColor=config["backColor"], lineWidth=1, interpolate = False,
                            fillColor=config["backColor"], autoLog=False, depth=1)

# ===========================================================================
#
#                        INSTRUCTION & FUNCTIONS
#
# ===========================================================================

# %% GENERATE INSTRUCTION
# ---------------------------------------------------------------------------

#task_text = "Please look at the circle. If you detect any brief change, \n\n press 1 for decrease (dimmer); press 2 for increase (brighter). \n\nPress any key to continue."
task_text = "Please keep your eyes on the center.\n\n If you notice a orientation change in the blinking dots, press 1. \n\nIf you are ready, press any key to continue."
instr_text = "Press any key to start."
break_text = "Please take a short break.\n\nPress the spacebar to continue."

end_run_text = "You've completed this run! Take a short break.\n\nThe next run will start soon."
end_end_text = "All done!\n\nPlease stay still and wait for the experimenter."

instr_stim = visual.TextStim(win=expWin, text=instr_text,
                             color="black", height=0.5*ppd, pos=(0,0),
                             autoLog=False)

# %% SHIFT FRAMES
# ---------------------------------------------------------------------------
def shift_frames(dir_factor):
    """
    Shift the moving frames for the current refresh rate

    dir_factor: 1 for right, -1 for left
    """
    frame_outer.pos = (frame_outer.pos[0] - dir_factor * config["shift_pix_per_frame"],
                           config["stim_pos"][1])
    frame_inner.pos = frame_outer.pos


def shift_frames_by_clock(dir_factor, time_elapsed, frame_default_pos):
    """
    Shift frames according to the time elapsed since trial onset

    time_elapsed: seconds since trial start
    dir_factor: initial direction (1 for right, -1 for left)
    frame_default_pos: initial (x, y) position tuple
    """

    one_way_duration = exp["reversal_time"] + exp["dot_duration"]
    n_one_way = int(time_elapsed / one_way_duration)

    # how far to shift
    time_left_to_shift = time_elapsed % one_way_duration
    if time_left_to_shift > exp["reversal_time"]:
        time_left_to_shift = exp["reversal_time"]
    if n_one_way % 2 == 1:
        time_left_to_shift = exp["reversal_time"] - time_left_to_shift

    shift_by_time = dir_factor * config["shift_pix_per_second"] * time_left_to_shift

    # move the frames
    frame_outer.pos = (frame_default_pos[0] + shift_by_time, frame_default_pos[1])
    frame_inner.pos = frame_outer.pos

# %% DRAW AND COLOR DOTS
# ---------------------------------------------------------------------------
def draw_dot(dot, draw_or_not = True):
    if FLASH_GRAB:
        for pie in dot:
            pie.autoDraw = draw_or_not
    else:
        dot.autoDraw = draw_or_not

def color_dot(dot, color):    
    if FLASH_GRAB:
        # 0, 2 -> color / 1, 3 -> opposite
        opposite = "black" if color == "white" else "white"
        if color == "red":
            opposite = "white"
        elif color == "white_red":
            color = "white"
            opposite = "red"
        for i, pie in enumerate(dot):
            if i % 2 == 0:  # 0, 2
                pie.fillColor = color
                pie.lineColor = color
            else:  # 1, 3
                pie.fillColor = opposite
                pie.lineColor = opposite
    else:
        dot.fillColor = color
        dot.lineColor = color

def update_dots(pair="upper", color="white", sets = dots):
    """
    Update the dot pair to be presented on the screen.

    pair: "upper" or "lower"
    color: "white" or "black"
    """
    if pair == "upper":
        color_dot(sets[0], color)
        draw_dot(sets[0], True)
        draw_dot(sets[1], False)
    elif pair == "lower":
        color_dot(sets[1], color)
        draw_dot(sets[1], True)
        draw_dot(sets[0], False)

# %% UPDATE DOTS (REVERSAL)
# ---------------------------------------------------------------------------
def update_dots_by_clock(time_elapsed, dot_color, dot_pos, dot_counter=0):
    """
    Update dots according to the time elapsed since trial onset
    """
    # config["duration_one_way"] = exp["reversal_time"] + exp["dot_duration"]
    n_one_way, time_left = divmod(time_elapsed, config["duration_one_way"])
    current_color = dot_color; current_pos = dot_pos
    
    if time_left > exp["reversal_time"]:
        # alternate per reversal
        if n_one_way % 2 == 1:
            # color (for the old stimulus)
            if expInfo['stimulus'] == "old":
                current_color = "black" if dot_color == "white" else "white" 
            # location
            current_pos = "upper" if dot_pos == "lower" else "lower"
            current_color = "black" if dot_color == "white" else "white"
        # alternate per frame; color (for the checkerboard)
        if FLASH_GRAB:
            dot_counter += 1
            if dot_counter % 1 == 0:            
                #current_color = "black" if dot_color == "white" else "white" 
                #current_color = "white" if dot_color == "white" else "black"
                if dot_color == "red":
                    current_color = "white_red"
                elif dot_color == "white_red":
                    current_color = "red"
        update_dots(current_pos, current_color)
    else:
        for dot in dots: draw_dot(dot, False) # hide dots
    return current_color, dot_counter

def update_markers_by_clock(time_elapsed, dot_color, dot_pos):
    """
    Update dots according to the time elapsed since trial onset
    """

    n_one_way, time_left = divmod(time_elapsed, config["duration_one_way"])
    current_color = dot_color; current_pos = dot_pos
    
    if time_left > exp["reversal_time"]:
        # alternate per reversal
        if n_one_way % 2 == 1:
            # color (for the old stimulus)
            # if expInfo['stimulus'] == "old":
            #     current_color = "black" if dot_color == "white" else "white" 
            # location
            current_pos = "upper" if dot_pos == "lower" else "lower"
        # alternate per frame; color (for the checkerboard)
        if FLASH_GRAB:
            current_color = "black" if dot_color == "white" else "white" 
        update_dots(current_pos, current_color, markers)
    else:
        for marker in markers: draw_dot(marker, False) # hide dots
    return current_color

# %% ADJUST MARKERS
# ---------------------------------------------------------------------------
def move_dots(distance):
    """
    Move dots from the screen center for a mouse response.
    """
    pos_list = [
        [config["stim_pos"][0] - distance, config["stim_pos"][1] + config["center_offset_y"]],
        [config["stim_pos"][0] + distance, config["stim_pos"][1] - config["center_offset_y"]],
    ]
    for i, new_pos in enumerate(pos_list):
        if FLASH_GRAB:
            for pie in dots[i]:
                pie.pos = new_pos
        else:
            dots[i].pos = new_pos
    for i, new_pos in enumerate(pos_list):
        if FLASH_GRAB:
            for pie in dots[i]:
                pie.pos = new_pos
        else:
            dots[i].pos = new_pos

def move_markers(distance):
    """Move markers from the screen center for a mouse response."""
    pos_list = [
        [config["mark_pos"][0] - distance, config["mark_pos"][1] + config["center_offset_y"]],
        [config["mark_pos"][0] + distance, config["mark_pos"][1] - config["center_offset_y"]],
    ]
    for i, new_pos in enumerate(pos_list):
        if FLASH_GRAB:
            for pie in markers[i]:
                pie.pos = new_pos
        else:
            markers[i].pos = new_pos

def update_markers(distance, dot_color):
    # adjust the position
    move_markers(distance)
    # OLD STIM
    if trial["InitialPosition"] == "upper":
        upper_col = trial["Start_col"]
        lower_col = "black" if upper_col == "white" else "white" 
    else:
        lower_col = trial["Start_col"]
        upper_col = "black" if lower_col == "white" else "white"

    # PLAIN
    if expInfo["stimulus"] == "plain":
        upper_col = dot_color
        lower_col = dot_color

    # CHECKERBOARD
    if FLASH_GRAB:
        # alternate the colors if it was not done by the dots
        if not dots[0][0].autoDraw:
            dot_color = "black" if dot_color == "white" else "white"
        for marker in markers:
            # 0, 2 -> color / 1, 3 -> opposite
            opposite = "black" if dot_color == "white" else "white"
            for i, pie in enumerate(marker):
                if i % 2 == 0:  # 0, 2
                    pie.fillColor = dot_color
                    pie.lineColor = dot_color
                else:  # 1, 3
                    pie.fillColor = opposite
                    pie.lineColor = opposite  
            draw_dot(marker, True)         
    else:
        markers[0].lineColor = upper_col
        markers[0].fillColor = upper_col
        markers[0].autoDraw = True
        markers[1].lineColor = lower_col
        markers[1].fillColor = lower_col
        markers[1].autoDraw = True
    return dot_color

# %% RUN TRIAL
# ---------------------------------------------------------------------------
def show_stimulus(trial, when_to_flip, contrast=False, tilt=True):
    """
    Run a single trial for a given trial condition
    """
    # Check the trial type
    trial_type = trial["ExpCondition"]
    dir_map = {"left": -1, "right": 1, "NA": 0}
    dir_factor = dir_map.get(trial["InitialDirection"], 0)
    dot_pos = trial["InitialPosition"]
    dot_color = "black" if np.random.randint(0, 2) == 1 else "white"
    
    if trial_type == "Perceptual":
        if trial["InducedPercept"] == "left":
            #perceived_distance = deg2pix(PRESCAN_AVG)
            perceived_distance = deg2pix(PRESCAN_LEFT)
        elif trial["InducedPercept"] == "right":
            #perceived_distance = -deg2pix(PRESCAN_AVG)
            perceived_distance = -deg2pix(PRESCAN_RIGHT)

    # Prepare the stimuli
    if trial_type == "Illusion":
        frame_default_pos = (config["stim_pos"][0] - 1/2 * config["frame_travel"] * dir_factor, config["stim_pos"][1])
        frame_outer.pos = frame_default_pos; frame_inner.pos = frame_default_pos
        frame_outer.autoDraw = True; frame_inner.autoDraw = True
        move_dots(0)
    # Prepare the stimuli           
    elif trial_type == "Physical":
        frame_outer.autoDraw = False; frame_inner.autoDraw = False
        move_dots(0)
    elif trial_type == "Perceptual":
        frame_outer.autoDraw = False; frame_inner.autoDraw = False
        move_dots(perceived_distance)

    # Stimulus Flags
    Resp = False
    Show = True
    dot_counter = 0

    # Calculate the wait duration
    wait_duration = when_to_flip - (exp_clock.getTime())
    when_to_flip += config["duration_one_block"]
    core.wait(wait_duration)

    # Log the trial info
    nt_label = trial["Trial"]
    el_tracker.sendMessage(f"TRIALID {nt_label}")
    el_tracker.sendCommand(f"record_status_message 'TRIAL number {nt_label}'")
    #el_tracker.sendCommand(draw_cmd) # custom display for eyelink host pc
    log_with_time(); logFile.write(f'Trial {nt_label} Start \n')

    # Reset the clock for resp
    expWin.recordFrameIntervals = True
    trial_clock = core.Clock(); trial_clock.reset()

    # # Set the time for contrast
    # if contrast:
    #     fixation_changed = False
    #     fixation_reset_time = None
    #     fix_change_time = np.random.uniform(0.3, 8) 
    #     fix_resp_window = 0
    #     cont_ans = 'none'

    # Set the time for stimulus change
    if tilt:
        rotated = False
        tilt_resp_window = 0
        #base_dot_color = dot_color

        min_gap = 2*3
        max_gap = 2*8
        min_one_way = 2
        max_one_way = int(config["duration_stimulus"] // config["duration_one_way"])
        tilt_events_1 = np.random.randint(min_one_way, min_one_way+max_gap)
        tilt_events_2 = np.random.randint(tilt_events_1+min_gap, tilt_events_1+max_gap)
        tilt_events = [tilt_events_1]
        if tilt_events_2 < max_one_way:
            tilt_events.append(tilt_events_2)
        tilt_events = set(tilt_events)
    
    # Show the stimulus
    while Show:
        # Update Stimuli
        time_elapsed = trial_clock.getTime()

        # Tilt Task
        if tilt:
            n_one_way, time_left = divmod(time_elapsed, config["duration_one_way"])
            n_one_way = int(n_one_way)
            # If the tilt event should be on but not yet rotated
            if n_one_way in tilt_events and not rotated:                                          
                # Rotate all dots
                for dot in dots:
                    rotate_checker(dot, 20)
                rotated = True
                # Flag for response window
                tilt_resp_window += 1
                # Log the tilt event onset
                trial[f"onset_{tilt_resp_window}"] = trial_clock.getTime()
                log_with_time(); logFile.write(f'Trial {nt_label} ROTATION ON at {time_elapsed:.3f}s\n')
                # Ready to accept response
                Resp = False 
                trial[f"resp_{tilt_resp_window}"] = 'Missed'
                trial[f"RT_{tilt_resp_window}"] = None
                trial[f"corr_{tilt_resp_window}"] = 0             
            # If the tilt event is over but still need to reset the dots
            elif n_one_way not in tilt_events and rotated: 
                for dot in dots:
                    rotate_checker(dot, 0)   
                rotated = False
        # Update stimulus and refresh the screen                    
        current_color, dot_counter = update_dots_by_clock(time_elapsed, dot_color, dot_pos, dot_counter)
        shift_frames_by_clock(dir_factor, time_elapsed, frame_default_pos) if trial_type == "Illusion" else None
        expWin.flip()

        # # Contrast Task
        # if contrast:
        #     # Fixation contrast on
        #     if not fixation_changed and time_elapsed >= fix_change_time:
        #         fix_resp_window += 1
        #         cont_ans = np.random.choice(['dec', 'inc'])
        #         fix_outer.color = config[f"fix_{cont_ans}"]
        #         trial[f"resp_ans_{fix_resp_window}"] = cont_ans
        #         fixation_changed = True
        #         fixation_reset_time = time_elapsed + 0.2  # 200ms
        #         log_with_time(); logFile.write(f'Trial {nt_label} Fixation Contrast ON ({cont_ans}) at {time_elapsed:.3f}s\n')
        #         Resp = False

        #     # Fixation contrast off
        #     if fixation_changed and fixation_reset_time and time_elapsed >= fixation_reset_time:
        #         fix_outer.color = config["fix_color"]
        #         fixation_changed = False
        #         fixation_reset_time = None
        #         log_with_time(); logFile.write(f'Trial {nt_label} Fixation Contrast OFF at {time_elapsed:.3f}s\n')
        #         fix_change_time += np.random.uniform(4, 8)
                
        # Check for response
        all_keys = event.getKeys()
        if '1' in all_keys or '2' in all_keys:
            first_key = all_keys[0]
            log_with_time(); logFile.write(f'Trial {nt_label} RESPONSE REPORTED at {time_elapsed:.3f}s\n')
            # # Contrast Task
            # if contrast and not Resp and (first_key == '1' or first_key == '2'):
            #     resp_cont = 'dec' if first_key == '1' else 'inc'
            #     trial[f"resp_cont_{fix_resp_window}"] = resp_cont
            #     trial[f"resp_RT_{fix_resp_window}"] = trial_clock.getTime()
            #     if cont_ans == resp_cont:
            #         trial[f"resp_cor_{fix_resp_window}"] = True
            #     el_tracker.sendMessage('TRIAL {nt_label}: response_saved')
            #     log_with_time(); logFile.write(f'Trial {nt_label} Response Saved ({resp_cont})\n')
            #     Resp = True

            ## Tilt task
            # If resp for current window not yet recorded
            if tilt and not Resp and (first_key == '1'):
                trial[f"clock_{tilt_resp_window}"] = trial_clock.getTime()
                # correct response: save RT
                if tilt_resp_window > 0:
                    trial[f"resp_{tilt_resp_window}"] = 'Hit'
                    trial[f"RT_{tilt_resp_window}"] = trial[f"clock_{tilt_resp_window}"] - trial[f"onset_{tilt_resp_window}"]
                    trial[f"corr_{tilt_resp_window}"] = 1
                    log_with_time(); logFile.write(f'Trial {nt_label} RESPONSE HIT; RT = {trial[f"RT_{tilt_resp_window}"]:.3f}s\n')
                # or record false alarm
                else:
                    trial[f"resp_{tilt_resp_window}"] = 'FalseAlarm'
                    trial[f"RT_{tilt_resp_window}"] = trial[f"clock_{tilt_resp_window}"]
                    trial[f"corr_{tilt_resp_window}"] = 0
                    log_with_time(); logFile.write(f'Trial {nt_label} RESPONSE FALSE ALARM; RT = {trial[f"RT_{tilt_resp_window}"]:.3f}s\n')                    
                Resp = True
        elif 'escape' in all_keys and trial_clock.getTime() > 1:
            expWin.close(); core.quit()
            
        # End of the block
        if trial_clock.getTime() > config["duration_stimulus"]:
            Show = False

    # Clear screen and events
    event.clearEvents()
    if tilt:   
        if rotated == True:
            for dot in dots:
                rotate_checker(dot, 0)   
            rotated = False          
    for dot in dots: draw_dot(dot, False)
    frame_outer.autoDraw = False; frame_inner.autoDraw = False
    expWin.recordFrameIntervals = False
    expWin.flip()
    el_tracker.sendMessage(f"STIM_OFF {nt_label}")
    log_with_time(); logFile.write(f'Trial {nt_label} Stimulus Off \n')

    # Stop tracker recording and log the trial end
    el_tracker.sendMessage(f'!V TRIAL_VAR condition {trial["InducedPercept"]}')
    el_tracker.sendMessage(f"TRIAL END {nt_label}")
    
    return trial, when_to_flip

def test_stimulus(trial):
    """
    Run a single trial to test the effect size
    """
    
    # Check the trial type
    trial_type = trial["ExpCondition"]
    dir_map = {"left": -1, "right": 1, "NA": 0}
    dir_factor = dir_map.get(trial["InitialDirection"], 0)
    dot_pos = trial["InitialPosition"]
    dot_color = "white"
    dot_counter = 0

    # Prepare the stimuli
    marker_dist = np.random.uniform(0,1*ppd) # randomize the marker placeholders
    move_dots(0)
    move_markers(marker_dist)

    # Stimulus Flags
    Resp = False

    # Log the trial info
    nt_label = trial["Trial"]
    el_tracker.sendMessage(f"TRIALID {nt_label}")
    el_tracker.sendCommand(f"record_status_message 'TRIAL number {nt_label}'")
    log_with_time(); logFile.write(f'Trial {nt_label} Start \n')
    #el_tracker.sendCommand(draw_cmd) # custom display for eyelink host pc

    # Reset the clock for resp
    expWin.recordFrameIntervals = True
    trial_clock = core.Clock(); trial_clock.reset()
    t_duration = config["duration_stimulus"]

    while not Resp:
        # Switch the stimulus
        if trial_type == "Illusion":
            frame_default_pos = (config["stim_pos"][0] - 1/2 * config["frame_travel"] * dir_factor, config["stim_pos"][1])
            frame_outer.pos = frame_default_pos; frame_inner.pos = frame_default_pos
            frame_outer.autoDraw = True; frame_inner.autoDraw = True
            for dot in dots: draw_dot(dot, True) # show dots
            for marker in markers: draw_dot(marker, False) # hide markers
        else:
            frame_outer.autoDraw = False; frame_inner.autoDraw = False
            for marker in markers: draw_dot(marker, True) # show markers
            for dot in dots: draw_dot(dot, False) # hide dots
        Show = True

        # Show the stimulus
        while Show:
            click_mouse = mouseResp.getPressed()
            wheel_dX, wheel_dY = mouseResp.getWheelRel()
            #click_mouse = mouseResp.getPressed()
            marker_dist += wheel_dY*2

            # Update Stimuli
            time_elapsed = trial_clock.getTime()
            if trial_type == "Illusion":
                shift_frames_by_clock(dir_factor, time_elapsed, frame_default_pos)
                dot_color, dot_counter = update_dots_by_clock(time_elapsed, dot_color, dot_pos, dot_counter)
            else:
                move_markers(marker_dist)
                dot_color = update_markers_by_clock(time_elapsed, dot_color, dot_pos)

            expWin.flip()
            el_tracker.sendMessage(f"STIM_ON {nt_label}")

            # Check for response
            all_keys = event.getKeys()
            if 'space' in all_keys or '4' in all_keys:
                trial['response_pix'] = marker_dist
                trial['response_dva'] = marker_dist / ppd                
                trial['response_time'] = trial_clock.getTime()
                el_tracker.sendMessage('TRIAL {nt_label}: response_saved')
                log_with_time(); logFile.write(f'Trial {nt_label} Response Saved \n')
                Resp = True
                Show = False
            elif '1' in all_keys:
                marker_dist += ppd/10
            elif '2' in all_keys:
                marker_dist -= ppd/10
            elif 'escape' in all_keys and trial_clock.getTime() > 1:
                expWin.close(); core.quit()

            # End of the block (illusion or perceptual)
            if trial_clock.getTime() > t_duration:
                trial_type = "Perceptual" if trial_type == "Illusion" else "Illusion"
                t_duration += config["duration_stimulus"]
                log_with_time(); logFile.write(f'Trial {nt_label} Switched to {trial_type} \n')
                Show = False


    # Clear screen and events
    event.clearEvents()
    for dot in dots: draw_dot(dot, False)
    for marker in markers: draw_dot(marker, False)
    fix_outer.color = "green"
    frame_outer.autoDraw = False; frame_inner.autoDraw = False
    expWin.recordFrameIntervals = False
    expWin.flip()
    core.wait(0.2)  # wait for a bit before the next trial
    fix_outer.color = config["fix_color"]
    expWin.flip()
    el_tracker.sendMessage(f"STIM_OFF {nt_label}")
    log_with_time(); logFile.write(f'Trial {nt_label} Stimulus Off \n')
    # Stop tracker recording and log the trial end
    el_tracker.sendMessage(f'!V TRIAL_VAR condition {trial["InducedPercept"]}')
    el_tracker.sendMessage(f"TRIAL END {nt_label}")
    log_with_time(); logFile.write(f'Trial {nt_label} Trial End \n')

    return trial


def show_until_response(trial, eye_used=1, fix_check=True, fix_valid = False):
    """
    Run a single trial until response
    """

    expWin.color = config["backColor"]
    expWin.flip()
    
    # Log the trial info
    nt_label = trial["Trial"]
    el_tracker.sendMessage(f"TRIALID {nt_label}")
    el_tracker.sendCommand(f"record_status_message 'TRIAL number {nt_label}'")
    #el_tracker.sendCommand(draw_cmd) # custom display for eyelink host pc

    # Validate the fixation
    if fix_check and EYELINK_CONNECTED:
        check_fixation(fix_duration=0.2, fix_radius_dva=1.5)  # Wait until fixation is held

    # Check the trial type
    dir_factor = -1 if trial["InitialDirection"] == "left" else 1
    dot_pos = trial["InitialPosition"]; dot_color = trial["Start_col"]

    # Prepare the stimuli
    frame_default_pos = (config["stim_pos"][0] - 1/2 * config["frame_travel"] * dir_factor, config["fix_pos"][1])
    frame_outer.pos = frame_default_pos; frame_inner.pos = frame_default_pos
    frame_outer.autoDraw = True; frame_inner.autoDraw = True
    marker_dist = np.random.uniform(0,2.5*ppd) # randomize the marker placeholders

    # Stimulus Flags
    # pauseFlag = False; updateFlag = False;
    Resp = False
    # Fixation Check
    new_sample = None; old_sample = None; fixation_lost = False
    fix_x, fix_y = config["fix_pos"]
    fixation_radius_px = ppd * 3
    fix_outer.fillColor = "red"; fix_outer.lineColor = "red"

    # Reset the clock
    trial_clock = core.Clock(); trial_clock.reset()
    expWin.recordFrameIntervals = True
    dot_counter=0

    # Loop until response
    while not Resp:
        click_mouse = mouseResp.getPressed()
        wheel_dX, wheel_dY = mouseResp.getWheelRel()
        marker_dist += wheel_dY*2

        # == EYELINK =======================================================================
        if fix_valid and EYELINK_CONNECTED:
            # Monitor fixation
            new_sample = el_tracker.getNewestSample() # Get the most recent sample
            if new_sample is None: # No sample
                continue
            elif old_sample is not None and new_sample.getTime() == old_sample.getTime(): # No new sample
                continue
            else:
                old_sample = new_sample

            # Get gaze coordinates
            if eye_used == 1 and new_sample.isRightSample():
                g_x, g_y = new_sample.getRightEye().getGaze()
            elif eye_used == 0 and new_sample.isLeftSample():
                g_x, g_y = new_sample.getLeftEye().getGaze()
            else: # No valid gaze
                continue

            # Center the gaze coordinates
            if g_x is not None:
                gx_centered = g_x - scn_width / 2
                gy_centered = (g_y - scn_height / 2) * -1
                # instr_stim.text = f'{gx_centered:.1f}, {gy_centered:.1f}'
                # instr_stim.draw()

            # Check if gaze has moved out of the fixation area
            if abs(gx_centered - fix_x) > fixation_radius_px and abs(gy_centered - fix_y) > fixation_radius_px:
                #fixation_lost = True
                #fix_outer.fillColor = "red"
                fix_outer.fillColor = config["fix_color"]
            else:
                fix_outer.fillColor = config["fix_color"] 
            # ====================================================================================
                
        # Update Stimuli
        time_elapsed = trial_clock.getTime()
        fix_outer.fillColor = config["fix_color"] if time_elapsed > exp["stabilize_time"] else "red"
        fix_outer.lineColor = fix_outer.fillColor
        current_color, dot_counter = update_dots_by_clock(time_elapsed, dot_color, dot_pos, dot_counter)
        dot_color = current_color if FLASH_GRAB else dot_color
        dot_color = update_markers(marker_dist, dot_color)
        shift_frames_by_clock(dir_factor, time_elapsed, frame_default_pos)

        # Draw and Flip
        frame_outer.draw(); frame_inner.draw()
        expWin.flip()

        # Check for response
        all_keys = event.getKeys()
        if time_elapsed > exp["stabilize_time"] and ('space' in all_keys or click_mouse[0]):
            resp_RT = trial_clock.getTime()
            trial['response_pix'] = 2 * marker_dist
            trial['response_dva'] = 2 * marker_dist / ppd
            trial['response_time'] = resp_RT
            el_tracker.sendMessage('TRIAL {nt_label}: response_saved')
            Resp = True # end of trial
        elif 'c' in all_keys and trial_clock.getTime() > 1:
            abort_trial()
            el_tracker.sendMessage(f'TRIAL {nt_label}: calibration_requested')
            expWin.flip()
            show_msg(expWin, "Calibrating eye tracker...\n\nPress ENTER to begin.")
            try:
                el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()            
            fixation_lost = True # will repeat the trial
            Resp = True # end the trial for now
        elif 'escape' in all_keys and trial_clock.getTime() > 1:
            expWin.close(); core.quit()

    # Catch fixation lost?
    if fixation_lost:
        trial["fix_valid"] = 0
        el_tracker.sendMessage(f'TRIAL {nt_label}: FIXATION LOST')
        TRIALS.append(copy.deepcopy(trial)) # retry
    else: trial["fix_valid"] = 1

    # Clear screen and events
    event.clearEvents()
    expWin.recordFrameIntervals = False
    for marker in markers: draw_dot(marker, False)
    for dot in dots: draw_dot(dot, False)
    frame_outer.autoDraw = False; frame_inner.autoDraw = False

    # End the trial -- show the ITI noise
    el_tracker.sendMessage(f'!V TRIAL_VAR condition {trial["InducedPercept"]}')
    el_tracker.sendMessage(f"TRIAL END {nt_label}")
    
    return trial

# %% VALIDATE FIXATION
# ---------------------------------------------------------------------------
def check_fixation(fix_duration=0.2, fix_radius_dva=1.5, eye_used=1):
    """
    The trial will only start if the participant maintains fixation for fix_duration.

    Parameters:
    - fix_duration: Required fixation duration before proceeding (in seconds)
    - fix_radius_dva: Allowed fixation radius in degrees of visual angle (DVA)
    """
    # Get fixation coordinates
    fix_x, fix_y = config["fix_pos"]
    fixation_radius_px = fix_radius_dva * ppd
    fix_outer.autoDraw = True; fix_inner.autoDraw = True

    # Declare flags and variables
    fixation_validated = False
    gaze_start_time = None
    old_sample = None; new_sample = None
    g_x = None; g_y = None

    # flip and get time
    expWin.flip()
    fix_clock = core.Clock()

    while not fixation_validated:

        # Get the most recent sample
        new_sample = el_tracker.getNewestSample()
        if new_sample is None:
            continue # No sample
        elif old_sample is not None and new_sample.getTime() == old_sample.getTime():
            continue # No new sample
        else:
            old_sample = new_sample # Update new sample

        # Get gaze coordinates
        if eye_used == 1 and new_sample.isRightSample():
            g_x, g_y = new_sample.getRightEye().getGaze()
        elif eye_used == 0 and new_sample.isLeftSample():
            g_x, g_y = new_sample.getLeftEye().getGaze()
        else: 
            continue # No valid gaze

        # Center the gaze coordinates
        if g_x is not None:
            gx_centered = g_x - scn_width / 2
            gy_centered = (g_y - scn_height / 2) * -1
            # instr_stim.text = f'{gx_centered:.1f}, {gy_centered:.1f}'
            # instr_stim.draw()

        # Check if gaze is within fixation area
        if abs(gx_centered - fix_x) < fixation_radius_px and abs(gy_centered - fix_y) < fixation_radius_px:
            if gaze_start_time is None:
                gaze_start_time = fix_clock.getTime()  # Start fixation timing
                fix_outer.fillColor = config["fix_color"]; expWin.flip()
            elif fix_clock.getTime() - gaze_start_time >= fix_duration:
                el_tracker.sendMessage("FIXATION_ACHIEVED")
                fixation_validated = True
        else:
            gaze_start_time = None  # Reset if gaze moves out
            fix_outer.fillColor = "red"; expWin.flip()

# %% EMERGENCY ESCAPE
# ---------------------------------------------------------------------------
def check_for_escape():
    """Check if the Escape key is pressed and exit the program."""
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        core.quit()

# ===========================================================================
#
#                                   RUN!
#
# ===========================================================================

# make sure the prescan summary is loaded
if expInfo['ses'] == "scan":
    show_msg(expWin, prescan_text, True)
    show_msg(expWin, task_text, True)


# %% PREPARE EYELINK
# ---------------------------------------------------------------------------

# Show tracker instructions
if EYELINK_CONNECTED and expInfo['run'] == "1":
    eyelink_msg = 'Calibrate the tracker.\n' + \
        'Please look at the center of the circle and stay still until the next one comes up.'
    show_msg(expWin, eyelink_msg)
    # Set up the camera and calibrate the tracker
    try:
        el_tracker.doTrackerSetup()
        el_tracker.setOfflineMode()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()
elif EYELINK_CONNECTED is False:
    eyelink_msg = 'Running in Dummy mode (with no tracker connection).\n' + \
        'Press any key to continue.'
    show_msg(expWin, eyelink_msg)
else:
    el_tracker.doTrackerSetup()    
    el_tracker.setOfflineMode()

# task instruction

# %% WAIT FOR THE TRIGGER
# ---------------------------------------------------------------------------

# Wait for the scanner to trigger
show_msg(expWin, "Waiting for scanner signal...", False)
event.waitKeys(keyList="5")

# Reset the global clock
exp_clock = core.Clock(); #exp_clock.reset()
logging.setDefaultClock(exp_clock)
t_start = exp_clock.getTime()

# Log and save the data
from datetime import datetime
def log_with_time():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.") + f"{datetime.now().microsecond // 1000:04d}"
    logFile.write(f"[{timestamp}] ")

log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logFile.write(f'Run {expInfo["run"]} started at {log_time}\n')
dataFile = []

# Refresh the screen and start tracker recording
fix_outer.autoDraw = True; fix_inner.autoDraw = True
expWin.color = config["backColor"]
expWin.flip(clearBuffer=True)
el_tracker.startRecording(1, 1, 1, 1)

# Wait for the dummy TRs
when_to_flip = exp["dummy_time"]

# %% RUN EXPERIMENT
# ---------------------------------------------------------------------------

# Loop over trials
for trial in TRIALS:

    # Ready for a trial
    tt_start = exp_clock.getTime()

    # Run a single trial
    if expInfo['ses'] == "scan":
        trial, when_to_flip = show_stimulus(trial, when_to_flip, contrast=False, tilt=True)
    else:
        test_stimulus(trial)
    #show_until_response(trial) # prescan at beaver

    # Log the trial duration
    trial_duration = exp_clock.getTime() - tt_start
    logFile.write(f'Trial Duration: {trial_duration:.3f} seconds\n')

    # Save the resonse
    executed_trial = trial.copy()
    dataFile.append(executed_trial)
    pd.DataFrame(dataFile).to_csv(dataName, index=False)

if expInfo['ses'] == "scan":
    core.wait(8) # wait for the last TR to finish


# %% WRAP UP
# ==============================================================================

# Log the experiment duration
t_end = exp_clock.getTime()
exp_duration = t_end - t_start
logFile.write(f'The experiment took {exp_duration:.3f} seconds')

# Save frame intervals
expWin.saveFrameIntervals(fileName=framesName)

# End of Experiment
if expInfo['ses'] == "scan":
    end_text = end_end_text if expInfo['run'] == exp['n_runs'] else end_run_text
    show_msg(expWin, end_text)
    core.wait(0.2)

    # Check the performance
    hits = 0; false_alarms = 0; missed = 0
    for d in dataFile:
        for key, value in d.items():
            if key.startswith('resp_'):
                if value == 'Hit':
                    hits += 1
                elif value == 'FalseAlarm':
                    false_alarms += 1
                elif value == 'Missed':
                    missed += 1
    show_msg(expWin, f"Hit Rates: {hits/(hits+missed)*100:.2f}%\nFalse Alarms: {false_alarms}")

# Analyze prescan results
elif expInfo['ses'] == "prescan":
    prescan_df = pd.DataFrame(dataFile)

    # Group and compute means
    grouped = prescan_df.groupby("InducedPercept")["response_dva"].mean()
    left_mean = abs(grouped.get("left", float("nan")))
    right_mean = abs(grouped.get("right", float("nan")))
    avg_abs = (left_mean + right_mean) / 2

    # Save the summary
    prescan_summary = pd.DataFrame([{
        "left": left_mean,
        "right": right_mean,
        "avg": avg_abs
    }])

    predataName = SUBDIR / f"sub-{sub_id}_{expInfo['sub_name']}_{expInfo['exp_name']}_{expInfo['ses']}_summary.csv"
    prescan_summary.to_csv(predataName, index=False)

    # Show the results
    result_text = f"Left mean: {left_mean:.2f}\nRight mean: {right_mean:.2f}\nAverage: {avg_abs:.2f}"
    show_msg(expWin, result_text, True)

# Close eyelink and transfer edf
el_tracker.stopRecording()
terminate_task()

# Close all
expWin.close()
try:
    core.quit()
except SystemExit:
    os._exit(0)

# %%
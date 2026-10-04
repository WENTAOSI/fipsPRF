"""
pylink.py -- STAND-IN / MOCK module
====================================
This is NOT the real SR Research pylink library. It exists only so that
`fips_stim.py` (which unconditionally does `import pylink`) can run on a
machine with no EyeLink hardware and no EyeLink Developer's Kit installed.

Every call here is a harmless no-op or returns a dummy value. It implements
just the subset of the pylink API that fips_stim.py and
EyeLinkCoreGraphicsPsychoPy.py touch, so you can preview/debug the visual
stimuli without installing the real SDK.

HOW TO USE
----------
Drop this file in the SAME folder as fips_stim.py (exp/code/). When you run
`python fips_stim.py` from that folder, Python puts the script's own
directory first on its import path, so this fake `pylink.py` is imported
instead of (or in lieu of) the real one.

IMPORTANT: this gives you NO real eye tracking. Messages/commands are just
printed to the console. Always choose "Eyelink Connected: no" in the dialog
box that pops up. Remove or rename this file later if you install the real
SR Research pylink package to run with actual hardware.
"""

import time

# ---------------------------------------------------------------------------
# Constants referenced by fips_stim.py / EyeLinkCoreGraphicsPsychoPy.py
# ---------------------------------------------------------------------------
TRIAL_OK = 0
TRIAL_ERROR = -1

CAL_TARG_BEEP = 1
CAL_GOOD_BEEP = 2
CAL_ERR_BEEP = 3
DC_TARG_BEEP = 4
DC_GOOD_BEEP = 5
DC_ERR_BEEP = 6

CR_HAIR_COLOR = 1
PUPIL_HAIR_COLOR = 2
PUPIL_BOX_COLOR = 3
SEARCH_LIMIT_BOX_COLOR = 4
MOUSE_CURSOR_COLOR = 5
SCREEN_OVERLAY_COLOR_1 = 6
SCREEN_OVERLAY_COLOR_2 = 7

IN_SETUP_MODE = 1

JUNK_KEY = -1
F1_KEY, F2_KEY, F3_KEY, F4_KEY, F5_KEY = 1, 2, 3, 4, 5
F6_KEY, F7_KEY, F8_KEY, F9_KEY, F10_KEY = 6, 7, 8, 9, 10
PAGE_UP, PAGE_DOWN = 11, 12
CURS_UP, CURS_DOWN, CURS_LEFT, CURS_RIGHT = 13, 14, 15, 16
ENTER_KEY = 17


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------
def msecDelay(ms):
    time.sleep(ms / 1000.0)


def pumpDelay(ms):
    time.sleep(ms / 1000.0)


def openGraphicsEx(genv):
    # Real pylink would hand calibration-drawing control to `genv`.
    # Nothing to do here since we never trigger real calibration.
    print("[pylink mock] openGraphicsEx called (no-op)")
    return 0


class KeyInput:
    """Minimal stand-in; only constructed, never really consumed in dummy mode."""
    def __init__(self, key, mod=0):
        self.key = key
        self.mod = mod


class EyeLinkCustomDisplay:
    """Base class EyeLinkCoreGraphicsPsychoPy subclasses. No behavior needed."""
    def __init__(self):
        pass


class _DummySample:
    """Returned by getNewestSample(); reports no valid gaze data."""
    def __init__(self):
        self._t = time.time()

    def getTime(self):
        return self._t

    def isRightSample(self):
        return False

    def isLeftSample(self):
        return False

    def getRightEye(self):
        return self

    def getLeftEye(self):
        return self

    def getGaze(self):
        return (0.0, 0.0)


class EyeLink:
    """
    Stand-in for pylink.EyeLink. Works the same whether you pass an IP
    address or None -- it never actually talks to any hardware.
    """
    def __init__(self, address=None):
        self.address = address
        self._connected = False  # always behaves like dummy/no-tracker mode
        self._recording = False
        print(f"[pylink mock] EyeLink({address!r}) created (dummy mode, no real tracker)")

    def isConnected(self):
        return self._connected

    def openDataFile(self, filename):
        print(f"[pylink mock] openDataFile({filename!r}) (no-op)")

    def close(self):
        print("[pylink mock] close() (no-op)")

    def setOfflineMode(self):
        pass

    def getTrackerVersionString(self):
        return "EYELINK MOCK 0"

    def sendCommand(self, cmd):
        pass

    def sendMessage(self, msg):
        pass

    def doTrackerSetup(self):
        print("[pylink mock] doTrackerSetup() called -- skipping (no real tracker)")

    def exitCalibration(self):
        pass

    def startRecording(self, *args):
        self._recording = True
        print("[pylink mock] startRecording() (no-op)")

    def stopRecording(self):
        self._recording = False

    def isRecording(self):
        return TRIAL_OK if self._recording else TRIAL_ERROR

    def closeDataFile(self):
        pass

    def receiveDataFile(self, src, dst):
        print(f"[pylink mock] receiveDataFile({src!r}, {dst!r}) (no-op, no file transferred)")

    def getNewestSample(self):
        return _DummySample()

    def getCurrentMode(self):
        return 0

    def readRequest(self, *args):
        pass

    def readReply(self):
        return "0"

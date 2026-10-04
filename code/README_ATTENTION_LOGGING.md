# Incremental attention-task logging

This version preserves the current preview mode, fixation-only baselines,
PsychoPy 2026.2.4 fixes, and all existing FIPS stimulus behavior.

The fixation-color attention log is now written incrementally.

As soon as the scanner run starts, this file is created:

    ../data/psycphys/logs/
    sub-XX_FIPS_scan_R_fixation_task.csv

The CSV contains:
- task_start
- every fixation color_change
- every accepted button_press

Each event is appended and flushed to disk immediately.

Therefore:
- a full run preserves the complete attention log
- pressing Escape midway preserves everything recorded up to that point
- stopping the run midway from PsychoPy should still leave the events that
  had already been flushed before the process was stopped

The termination routine also explicitly flushes/closes the attention log
before shutting down EyeLink and PsychoPy.

Columns:
    event_type
    run_time_s
    scheduled_time_s
    color
    key

This version still records raw attention events only. It does not yet label
hits, misses, false alarms, or reaction times.

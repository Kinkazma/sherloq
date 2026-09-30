# ELA sliders: manual input takes precedence

I fixed a case where an automatic ELA estimate already in progress could replace
the four displayed settings after a user interacted with a slider. A key or
wheel movement at a limit, for example, may not change the numerical value or
emit the signals previously used to switch to manual mode.

The ELA biomes panel and Complete Automatic Analysis now take manual control
before Qt processes a left click, wheel movement or slider navigation key.
An obsolete automatic response cannot overwrite that manual choice. Focus, Tab
and right clicks do not switch modes; explicitly selecting an automatic profile
still works.

The reproduced failure changed `[0, 990, 50, 50]` to `[78, 980, 80, 10]` after
pressing Home while the first slider was already at zero (internal control units
are tenths). The corrected version keeps `[0, 990, 50, 50]` and selects Manual.
Ordinary drags already passed before this fix; the reproduced race does not
establish that every reported slider symptom had the same cause.

Shadow and highlight deviation thresholds remain independent. Changing a
histogram bound changes their shared reference and can legitimately change both
maps without moving the other sliders. Formulas, numerical profiles, caches and
CPU/GPU algorithms are unchanged.

## Update an installed RC1

The six-part RC1 archive remains unchanged. Close SHERLOQ and, from a freshly
downloaded or updated repository folder, run with Python 3.11 or newer:

```sh
python3 macos/apply_rc1_updates.py "/path/to/SHERLOQ-installation"
```

For a fresh installation, run its `restore_installation.py` first. This cumulative
update accepts original RC1 files and installations carrying earlier fixes.
It now also includes the [PatchMatch memory correction](PATCHMATCH-MEMORY.md),
validates 65 source/build/inventory files before changing them, and saves originals under
`.updates/rc1-native-20260930/files/`. Earlier update backups are retained.
Unknown local modifications are refused. Already updated files are left alone;
the earlier `apply_whole_image_fix.py` command also applies this cumulative update
when run from the current repository. Reopen the application to load the new code.

## Validation

Export the repository with `macos/stage_source.py` using the
[source-development instructions](../README.md#source-development). From that
export, with the application's dependencies installed, run:

```sh
QT_QPA_PLATFORM=cocoa python tests/ela-slider-independence/check.py
QT_QPA_PLATFORM=cocoa python tests/whole-image/check.py
```

The slider test covers 64 Cocoa cases in the two panels, using French labels:
real drags from three built-in profiles and one saved profile, preservation of
the other three values and control geometry, and automatic worker responses
delayed until after keyboard, wheel, groove and handle interactions at limits.
It also checks that automatic profile selection remains available afterwards.

The input is the synthetic `tests/sample.jpg`; no clone models run. The script
writes `results.json` and an interface capture beside itself. Additional native
regression checks on a private reference image passed separately, including
stability of the opposite energy class; that private corpus is not distributed.

The current cumulative update also includes the [validated adaptive-memory paths](ADAPTIVE-MEMORY.md) and rebuilds their native bridge.

# Run an automatic analysis on the whole image

Complete Automatic Analysis and Automatic Clone Search still detect subimages
when opened. I added **Run on whole image** (**Exécuter sur toute l’image** in
French) so an incorrect detection does not prevent analysing the complete input.

The button is available during detection, during analysis and after a detection
error. It clears detected regions and exclusions, selects the full image and
restarts the analysis with the current settings. Late results from the previous
run are discarded. **Detect subimages** returns to automatic region detection.
In Complete Automatic Analysis, ELA also restarts for the new scope.

## Install the correction

The repository source includes the fix. The six-part **RC1 installation archive
is unchanged**. For an existing RC1 installation, close SHERLOQ, obtain the
current repository with **Code → Download ZIP** (or update your checkout), and
open Terminal in that repository folder. Run with Python 3.11 or newer:

```sh
python3 macos/apply_rc1_updates.py "/path/to/SHERLOQ-installation"
```

Replace the quoted path with your extracted installation folder. For a fresh
installation, run `restore_installation.py` there **before** applying this update.
Then reopen its `build/SHERLOQ.app`, or run `venv/bin/python app_start.py` from the
installation folder. No environment, model download or app-bundle rebuild is needed. The current updater rebuilds the native bridge; see [adaptive-memory update details](ADAPTIVE-MEMORY.md).

The updater verifies all 65 source/build/inventory files before replacement,
keeps the original files under `.updates/rc1-native-20260930/files/`, and refuses
unrecognized local changes. Repeating the command leaves already updated files
alone. Its baseline hashes were checked against the actual RC1 archive. The cumulative
update also includes the [ELA manual-control correction](ELA-SLIDERS.md) and
[PatchMatch memory correction](PATCHMATCH-MEMORY.md), and
accepts installations that already received the earlier whole-image update.

## Reproduce the regression check

Export the source using `macos/stage_source.py` as described in the
[installation guide](../README.md#source-development). In that exported layout,
using an environment with the application's dependencies, run:

```sh
QT_QPA_PLATFORM=cocoa python tests/whole-image/check.py
```

The test uses synthetic pixels, real Qt jobs and real ELA calculations. Clone
outputs are explicitly lightweight fixtures to isolate scope and cancellation
behaviour; this is not a new accuracy evaluation of the research models.

Both panels are checked for startup detection, cleared exclusions, empty or
failed detection, early clicks and return to automatic detection. Six worker
paths are deliberately completed after cancellation to check that obsolete
results are ignored. An obsolete detection error is also rejected. The two
injected detector-failure messages are expected test inputs.

The check writes `results.json` and a synthetic interface capture next to itself.

The current cumulative update also includes the [validated adaptive-memory paths](ADAPTIVE-MEMORY.md) and rebuilds their native bridge.

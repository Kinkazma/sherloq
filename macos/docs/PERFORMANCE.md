# Recorded performance measurements

I recorded these before/after profiles during development on 27 September 2026 on my Apple Silicon Mac. They compare the saved pre-change tool implementations with the implementations at the time of each test. They were not rerun for this documentation update and are not a cross-machine benchmark of the complete release.

The [measurement records](PERFORMANCE-RECORDS.json) retain the original numeric fields and a SHA-256 of each original JSON record. Test images and private working directories are not included. The records support the figures below, but do not by themselves constitute a fully reproducible public benchmark dataset.

| Case | Operation and scope | Before | After |
| --- | --- | ---: | ---: |
| F10 — Channel Histogram | Same 20 MP JPEG; isolated parent process; QApplication construction excluded, widget show/draw included | 14.857729 s | 0.493100 s |
| F10 — Channel Histogram | Peak parent RSS in that run | 528,793,600 bytes | 390,217,728 bytes |
| F16 — Frequency Split | Initial 20 MP panel ready; default split/smoothing; GUI update included | 10.805265 s | 3.163923 s |
| F16 — Frequency Split | Three successive threshold changes; GUI update and debounce included | 10.032 / 9.963 / 9.898 s | 0.869 / 0.846 / 0.818 s |
| F19 — PCA Projection | Initial 20 MP panel ready | 2.433865 s | 1.147042 s |
| F19 — PCA Projection | Peak process RSS over the measured sequence | 5,288,624,128 bytes | 3,009,069,056 bytes |

GB in the README means decimal billions of bytes. Peak memory is process RSS for the measured session, not an allocation attributed solely to one algorithm.

## What changed and what the figures do not say

For histograms, RGB triplets are encoded as integers and counted without sorting a large array of triplets; histogram counts and distinct-colour counts are retained while display settings change. Both recorded runs found 348,849 distinct colours.

Frequency Split retains its transform and results, uses explicit spatial FFT axes, and separates expensive analysis changes from display-only changes. Display filtering was already inexpensive: in these records its three updates changed from 0.017/0.022/0.007 s to 0.024/0.027/0.015 s. I do not present those updates as an improvement.

PCA avoids retaining unnecessary three-channel gray images and defers some work until a view is requested. This reduces initial time and peak RSS in the recorded sequence, but the first visits to some deferred views are slower. For example, the first recorded distance-view change increased from 0.047 s to 0.147 s. Repeated-view caching and initial computation must be distinguished.

Background processing, caching, bounded memory and selected native kernels improve particular workflows. They do not imply universal CPU/GPU equivalence, an application-wide acceleration factor or greater forensic accuracy. Small images, cold starts, different parameters and other machines can behave differently.

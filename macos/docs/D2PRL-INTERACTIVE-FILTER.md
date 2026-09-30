# D2PRL — interactive region-size filter

I added a slider and synchronized numeric field to D2PRL in AI Clone Detection.
They select the minimum connected-region size from **0 to 5,000 pixels on the
448 × 448 model grid**, with **500** as the default. These are model-grid pixels,
not pixels of the original photograph.

- **500** preserves the original postprocessing exactly.
- **0** disables small-component removal only. The probability threshold and
  the 50 × 50 source/target majority filter remain unchanged.
- Regions strictly smaller than the selected minimum are removed. Each analysis
  zone keeps its own connected components before results are combined at the
  original image coordinates.

Changing the value refilters the raw maps already held in memory. It does not
rerun neural inference or perform a calibration. The mask, source/target maps,
overlay and export follow the selected value; **Map** continues to show the raw
probability map. The D2PRL overlay now colours only the retained mask.

Rapid changes are grouped over 80 ms, and obsolete results cannot replace the
latest choice. Export is disabled until filtering completes. Changing the
backend, zones or analysis invalidates pending filter work. Export metadata
records the minimum globally and for each zone.

A recorded cached 1254 × 1254 image took **6.6–10.8 ms** to refilter at tested
values of 0, 100, 403 and 500. These are individual computation times, excluding
event grouping and drawing, not end-to-end latency guarantees. Lowering the
minimum can reveal more candidates; it does not establish that they are genuine
retouches. The [earlier negative examples](D2PRL-INTEGRATION.md#two-additional-image-tests)
remain results measured with the original default of 500.

## Validation and installation

The public-export checks repeat twelve original postprocessing comparisons and
independent-zone checks, plus six region-size boundaries and Qt tests for slider/
numeric synchronization, rapid changes, invalidation, overlay masking and no
inference calls. The default remains exact. The original image timing record is
kept separately and was not repeated on the private TIFF for publication.

- [Delivered reference and timing record](../tests/d2prl/filter-ui.json)
- [Independent public checks](D2PRL-FILTER-PUBLIC-CHECKS.json)
- [Source and translation manifest](D2PRL-FILTER-SOURCE-MANIFEST.json)
- [Installer checks from six published baselines](../tests/rc1-updater/d2prl-filter-installation-results.json)

Use the [cumulative RC1 updater](ADAPTIVE-MEMORY.md#install-the-cumulative-rc1-update)
after closing SHERLOQ. It accepts the previous D2PRL source version and preserves
its backups. No new checkpoint is needed for this filter. The existing
[D2PRL external model requirements](D2PRL-INTEGRATION.md#installation-and-model-files)
still apply to neural inference.

From the staged source layout:

```sh
python tests/d2prl/contracts.py
QT_QPA_PLATFORM=offscreen python tests/d2prl/filter_ui.py
```

These two tests use synthetic raw maps and do not require a model checkpoint.

# Optional new-feature highlights

I made the red, green and blue outlines for new features optional. They are hidden
by default. Enable **View > Highlight new features** to display them; in French,
use **Affichage > Surbrillance des nouveautés**. The application remembers the
choice between sessions.

The option applies globally to the tool tree and favorites, combo-box selections
and popup lists, tabs and internal controls. Open panels repaint immediately, and
new panels follow the same setting. Switching the option does not rerun an
analysis, change parameter values, resize controls or change the image framing.
Scientific result contours and keyboard focus indicators are unaffected.

After installing the [cumulative update](../README.md#apply-the-cumulative-rc1-updates),
restart SHERLOQ once to load the new menu. Subsequent toggles take effect immediately.

The Qt offscreen check covers all three colours, exact restoration of the original
pixels after disabling the option, unchanged geometry and values, newly opened
panels, persistence and live French/English translation. This is a display check;
no detector or scientific result was changed.

[Source merge and preservation inventory](NEW-FEATURE-HIGHLIGHTS-SOURCE.json) ·
[UI test](../tests/extension-highlights/check.py).

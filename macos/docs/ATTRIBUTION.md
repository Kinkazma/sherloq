# Attribution and distribution scope

SHERLOQ: Guido Bartoli and the contributors credited in the original source.
The original GPLv3 text is retained in `source/LICENSE`. Existing author names,
notices and upstream contact addresses are not anonymized. I publish this contribution under my GitHub account, Kinkazma, on 29 September 2026.
The source/ paths below refer to the staged native installation layout.

The following is an inventory of license files inspected in this snapshot,
not a statement that all optional dependencies share SHERLOQ's license.

| Included component | Evidence retained | Source / attribution |
| --- | --- | --- |
| SHERLOQ application | source/LICENSE and file notices | GuidoBartoli/sherloq, base revision in README |
| PyExifTool wrapper only | COPYING.BSD, COPYING.GPL and file notices | upstream bundled wrapper; ExifTool executables excluded |
| IPOL PatchMatch adaptation | GPLv3.txt, UPSTREAM.txt, README.md | Thibaud Ehret, IPOL article 213; source archive hash in README |
| VLFeat | src/vlfeat/COPYING | Andrea Vedaldi, Brian Fulkerson, VLFeat team; BSD terms retained |
| Noisesniffer | LICENSE, PROVENANCE.md | IPOL article 462; Apache-2.0, source archive hash retained |
| ZERO | LICENSE and source notices | Quentin Bammey / IPOL; GNU AGPLv3, separate terms retained |
| XFeat | LICENSE, PROVENANCE.md | verlab/accelerated_features, revision e92685f57f8318b18725c5c8c0bd28c7fe188d9a; Apache-2.0 |
| LightGlue subset | LICENSE, PROVENANCE.md, individual headers | cvg/LightGlue, eb42fee2d71449efb0aa5c10549752b5d75384d8; Apache-2.0 plus ALIKED's retained BSD header |

The optional ZERO source retains AGPL obligations; it must not be described as a
uniformly GPL-only component or deployed as a network service without reviewing
those terms. The Git source tree contains source code; the Release separately contains the
complete installed runtime and binaries with their supplied notices.

## Permission and full installation

The contributor confirms having obtained the necessary permissions by email for
this handover. This candidate relies on that confirmation; correspondence is not
redistributed and no further documentary check is required for this task.
Original component notices and license texts remain unchanged, including
nonprofit restrictions, AGPL terms and nested component terms. This document does
not rewrite those notices or assert a blanket license for third-party components.

All present research sources are included, including Noiseprint, TruFor, Adaptive
CFA, CAT-Net, SAFIRE, AdaIFL, FOCAL and the full LightGlue implementation. Their
existing per-file credits and available license texts accompany the sources.
The companion installation ZIP includes all locally present pretrained weights,
Python runtimes and native dependencies. Absent checkpoints remain absent and
are explicitly recorded; no replacement model is trained or downloaded.

Personal photographs, application settings, logs, screenshots of the workstation,
private Git/chat history, email and website/NAS data are excluded. Tests use
synthetic inputs. This does not remove installed analysis components.

## D2PRL native adapter — 30 September 2026

D2PRL inference is adapted from [byc33/D2PRL](https://github.com/byc33/D2PRL/tree/a4314b614ea3186b4fac98e9e96939e37f275fc5),
with the original file notices retained. The additional
[nPr0nn/D2PRLu reference](https://github.com/nPr0nn/D2PRLu/tree/70b804a16575fe160695ec6c4740e6f10337d3b1)
does not replace the inference implementation. Its Apache-2.0 license does not
establish distribution rights for the original checkpoint. The checkpoint stays
external to this source update and is absent from the frozen RC1 archive.
[Runtime requirements, protocol and measurements](D2PRL-INTEGRATION.md).

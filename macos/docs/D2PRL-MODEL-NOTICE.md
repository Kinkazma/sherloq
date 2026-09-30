# D2PRL checkpoint — provenance and terms

This separate GitHub Release asset is the final inference checkpoint used by
my native D2PRL adapter. It is mirrored unchanged and is not an original model
trained by this fork.

- File: `d2prl.pth`, 540,533,481 bytes.
- SHA256: `2749c7436169ce689deaeb197ce5dae3d1a4533999833928168ec0b0d703df36`.
- Original implementation: [byc33/D2PRL](https://github.com/byc33/D2PRL), pinned
  for this adapter at `a4314b614ea3186b4fac98e9e96939e37f275fc5`.
- Research credit: Yuanman Li, Yingjie He, Changsheng Chen, Li Dong, Bin Li,
  Jiantao Zhou and Xia Li, “Image Copy-Move Forgery Detection via Deep PatchMatch
  and Pairwise Ranking Learning,” IEEE Transactions on Image Processing,
  volume 34, pages 425–440, 2025.

The original repository publicly links pretrained weights. As inspected on
30 September 2026, it does not provide a separate license statement for this
checkpoint. Public availability and the permission to share a file with a
recipient do not themselves state a general redistribution license. This mirror
does not grant additional rights or claim that the checkpoint is covered by
SHERLOQ's GPLv3 license.

The [D2PRLu reference implementation](https://github.com/nPr0nn/D2PRLu) has an
Apache-2.0 notice; that notice is not applied here to the original checkpoint.
Original authors retain their rights. No private correspondence or private
file-sharing URL is included in this publication.

OSN is used for training and is not needed by the final inference model. Only
the final D2PRL checkpoint is mirrored in this supplement. The frozen RC1 ZIP
and its six checksummed parts are unchanged.

[Installation and verification](D2PRL-INTEGRATION.md#installation-and-model-files).

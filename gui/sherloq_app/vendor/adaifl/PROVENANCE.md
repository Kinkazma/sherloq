# AdaIFL

Source LMIAPC/AdaIFL at54999594a97e13fec4eceb15421a16f7ff29cfed, MIT.
SAM and ModuleFormer attributions in source retained; ForensicHub not required.
Local change in fi_attn.py: generate tie-breaking noise with a common CPU random
stream, then transfer to the active device. Inference seed reset in core/adaifl.py.
Other model computations and published PIL1024² preprocessing preserved.
The checkpoint loads strictly after removal of the DataParallel module prefix.
No automatic downloads, no training or outputs in the upstream repository.

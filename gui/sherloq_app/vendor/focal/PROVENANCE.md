# FOCAL

Source HighwayWu/FOCAL at2bebdc9396b6d9f0145d6532d7749ee992426c0e, MIT;
HRNet Microsoft notice and nested Meta SAM Apache2 licence retained.
Local changes: package-qualified imports, no sys.path additions in models/vit.py.
All active inference parameters must match strictly, replacing upstream's silent
partial load. The released fc.weight/fc.bias heads are absent from both published
feature extractors: their names and shapes are validated, then explicitly omitted.
KMeans keeps the published library default seed 123 (not the global torch seed). Model entry point and published inference are ported in core/focal.py;
upstream main.py, which deletes/recreates output directories, is not executed.
External weight licence uncertainty is documented in the user research report;
weights remain in the private installation, outside the source contribution kit.

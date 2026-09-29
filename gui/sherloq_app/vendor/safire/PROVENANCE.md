# SAFIRE

Source mjkwon2021/SAFIRE, revision ec2af6af8c789fb6359493a3e352f89c6fd8107f.
README announces Apache2 code and CC-BY4.0 weights; Meta SAM notices preserved.
Apache2 text copied from the equivalent SAM licence included in FOCAL's pinned
source distribution. No weights redistributed in the source contribution kit.

Changes: package-qualified imports; unused training imports removed from
networks/safire_model.py; zeros_like replaces CUDA-only FFT mask allocation.
Full SAFIRE checkpoint is strictly loaded after constructing the architecture
with no partial SAM checkpoint. All learned parameters covered.
core/safire.py streams batches16, retains low-resolution proposals for subsequent
regrouping, uses fixed-seed CPU clustering, guards empty regions/clusters and
all-noise DBSCAN. Reference predictor retained for tests and provenance.

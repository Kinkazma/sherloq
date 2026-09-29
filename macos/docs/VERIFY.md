# Verify, reconstruct and run

Use Python 3.11+. Verify the external ZIP checksum before extracting:

```sh
shasum -a 256 -c SHERLOQ-native-candidate.zip.sha256
unzip SHERLOQ-native-candidate.zip
cd SHERLOQ-native-candidate
python3 tools/verify.py
```

Reconstruct the exported application without downloading any history:

```sh
cp -R reference reconstructed
cd reconstructed
git apply --check ../patches/native.patch
git apply ../patches/native.patch
cd ..
python3 tools/verify.py --reconstructed reconstructed
```

Alternatively apply the same patch in an upstream checkout at the exact base.
Unrelated upstream assets are neither removed nor certified by this patch.

To create a fresh environment (network and compatible wheels required):

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock
.venv/bin/python tests/public/synthetic.py
.venv/bin/python tests/public/startup.py
cd source
../.venv/bin/python -m gui.sherloq_app
```

The lock is a version inventory, without wheel hashes. Do not treat a test using
an existing environment as proof that fresh installation succeeded. The companion full installation ZIP supplies installed external programs and
research components. No automatic weight
installation is part of this procedure.

Optional native rebuild, from the kit root on macOS:

```sh
mkdir -p native/runtime
.venv/bin/python packaging/build_patchmatch.py
.venv/bin/python packaging/build_zero.py
```

Compiler output is local and excluded from the distributed source ZIP. See the
adjacent candidate validation report for checks actually performed. Source ZIP
byte reproduction is done by the separately supplied reviewed exporter and input
allowlist; it refuses changed inputs. Never use the historical broad exporter.

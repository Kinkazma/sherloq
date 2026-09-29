"""Build ZERO reference and deterministic parallel CPU kernels on macOS ARM64."""
from pathlib import Path
import subprocess,hashlib,json
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'source/gui/sherloq_app/vendor/zero';BUILD=ROOT/'build/zero';BUILD.mkdir(parents=True,exist_ok=True)
flags=['-O3','-std=c99','-fPIC','-isysroot',subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()]
subprocess.run(['clang',*flags,'-Dzero=sherloq_reference_zero','-Dcompute_grid_votes_per_pixel=sherloq_reference_votes','-c',str(SRC/'zero.c'),'-o',str(BUILD/'reference.o')],check=True)
subprocess.run(['clang',*flags,'-c',str(SRC/'parallel.c'),'-o',str(BUILD/'parallel.o')],check=True)
dest=ROOT/'native/runtime/libsherloq_zero.dylib';subprocess.run(['clang','-dynamiclib',str(BUILD/'reference.o'),str(BUILD/'parallel.o'),'-o',str(dest),'-install_name','@rpath/'+dest.name],check=True);subprocess.run(['codesign','--force','--sign','-',str(dest)],check=True)
print(json.dumps(dict(binary=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))

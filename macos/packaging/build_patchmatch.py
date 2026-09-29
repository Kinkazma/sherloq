"""Compile pinned IPOL dense descriptors for ARM64, without image codecs/CUDA."""
from pathlib import Path
import subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source/gui/sherloq_app/vendor/patchmatch';BUILD=ROOT/'build/patchmatch';BUILD.mkdir(parents=True,exist_ok=True)
DEST=ROOT/'native/runtime/libsherloq_patchmatch.dylib'
STAGED=DEST.with_suffix('.new.dylib')
sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
common=['-O3','-fPIC','-isysroot',sdk,'-DVL_DISABLE_AVX','-DVL_DISABLE_SSE2','-I'+str(SRC/'src/vlfeat')]
files=[SRC/'src/vlfeat/vl'/f'{n}.c' for n in ('host','generic','dsift','scalespace','sift','random','imopv','imopv_sse2')]
files[2]=SRC/'dsift_gpu.c'  # Includes the unmodified VLFeat dsift.c and GPU preparation helpers.
files += [SRC/'src/Utilities/FeatManager'/f'{n}.cpp' for n in ('zMManager','siftManager')]+[SRC/'bridge.cpp']
objects=[]
for i,p in enumerate(files):
 obj=BUILD/f'{i}.o';compiler='clang' if p.suffix=='.c' else 'clang++';standard='-std=c99' if p.suffix=='.c' else '-std=c++11'
 subprocess.run([compiler,standard,*common,'-c',str(p),'-o',str(obj)],check=True);objects.append(str(obj))
for name in ('metal_zernike','metal_sift'):
 obj=BUILD/(name+'.o')
 subprocess.run(['clang++','-std=c++11',*common,'-fobjc-arc','-c',str(SRC/(name+'.mm')),'-o',str(obj)],check=True);objects.append(str(obj))
subprocess.run(['clang++','-dynamiclib',*objects,'-framework','Foundation','-framework','Metal','-o',str(STAGED),'-install_name','@rpath/'+DEST.name],check=True)
subprocess.run(['codesign','--force','--sign','-',str(STAGED)],check=True)
STAGED.replace(DEST)  # Atomic replacement leaves the running application's mapped inode intact.
print(json.dumps(dict(binary=str(DEST),sha256=hashlib.sha256(DEST.read_bytes()).hexdigest())))

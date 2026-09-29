from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'source/gui/sherloq_app/vendor/patchmatch';OUT=Path(__file__).parent
sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
flags=['-O3','-fPIC','-isysroot',sdk,'-DVL_DISABLE_AVX','-DVL_DISABLE_SSE2','-I'+str(SRC/'src/vlfeat')]
subprocess.run(['clang','-std=c99',*flags,'-c',str(SRC/'dsift_gpu.c'),'-o',str(OUT/'dsift.o')],check=True)
subprocess.run(['clang++','-std=c++11',*flags,'-c',str(SRC/'bridge.cpp'),'-o',str(OUT/'bridge.o')],check=True)
subprocess.run(['clang++','-std=c++17','-fobjc-arc',*flags,'-c',str(SRC/'metal_stream.mm'),'-o',str(OUT/'metal_stream.o')],check=True)
objects=[str(ROOT/'build/patchmatch'/f'{i}.o') for i in range(10) if i!=2]+[str(OUT/'dsift.o'),str(OUT/'bridge.o'),str(OUT/'metal_stream.o')]+[str(ROOT/'build/patchmatch'/f'{n}.o') for n in ('metal_zernike','metal_sift')]
subprocess.run(['clang++','-dynamiclib',*objects,'-framework','Foundation','-framework','Metal','-o',str(OUT/'candidate.next.dylib')],check=True)
subprocess.run(['codesign','--force','--sign','-',str(OUT/'candidate.next.dylib')],check=True)

(OUT/'candidate.next.dylib').replace(OUT/'candidate.dylib')

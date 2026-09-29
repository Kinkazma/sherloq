from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'source/gui/sherloq_app/vendor/patchmatch';OUT=Path(__file__).parent
sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
subprocess.run(['clang++','-std=c++11','-O3','-fPIC','-isysroot',sdk,'-I'+str(SRC),'-I'+str(SRC/'src/vlfeat'),'-c',str(ROOT/'tests/dense-streaming/bridge_before.cpp'),'-o',str(OUT/'legacy_bridge.o')],check=True)
objects=[str(ROOT/'build/patchmatch'/f'{i}.o') for i in range(10)]+[str(OUT/'legacy_bridge.o')]+[str(ROOT/'build/patchmatch'/f'{n}.o') for n in ('metal_zernike','metal_sift')]
subprocess.run(['clang++','-dynamiclib',*objects,'-framework','Foundation','-framework','Metal','-o',str(OUT/'legacy_bridge.dylib')],check=True)
subprocess.run(['codesign','--force','--sign','-',str(OUT/'legacy_bridge.dylib')],check=True)

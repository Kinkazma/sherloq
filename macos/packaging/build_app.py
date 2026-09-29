"""Build an ARM64 launcher for this installation; never replace Applications."""
import argparse,plistlib,subprocess,sys,shutil,platform
from pathlib import Path
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if sys.platform!='darwin' or platform.machine()!='arm64' or sys.version_info[:2]!=(3,11):
        raise SystemExit('Build with this installation’s ARM64 Python 3.11 on macOS.')
    root=Path(__file__).resolve().parents[1]
    if '\n' in str(root):raise ValueError('Installation path must not contain a newline.')
    app=(args.output or root/'build/SHERLOQ.app').resolve()/'Contents'
    if app.parent==Path('/Applications/SHERLOQ.app'):
        raise SystemExit('Build and validate in a staging directory before installation.')
    (app/'MacOS').mkdir(parents=True,exist_ok=True);(app/'Resources').mkdir(exist_ok=True)
    base=Path(sys.base_prefix)
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    binary=app/'MacOS/SHERLOQ'
    subprocess.run(['clang','-arch','arm64','-isysroot',sdk,'-I'+str(base/'include/python3.11'),str(root/'packaging/launcher.c'),'-L'+str(base/'lib'),'-lpython3.11','-Wl,-rpath,'+str(base/'lib'),'-o',str(binary)],check=True)
    # The standalone Python's install id contains its installation directory.
    # Only rewrite our new launcher; leave the running private runtime alone.
    libraries=subprocess.check_output(['otool','-L',str(binary)],text=True)
    for line in libraries.splitlines()[1:]:
        library=line.strip().split(' (')[0]
        if library.endswith('/libpython3.11.dylib'):
            subprocess.run(['install_name_tool','-change',library,'@rpath/libpython3.11.dylib',str(binary)],check=True)
    runtime=app/'Resources/runtime'
    if runtime.is_symlink():runtime.unlink()
    elif runtime.exists():raise FileExistsError(runtime)
    (app/'Resources/install-root.txt').write_text(str(root)+'\n')
    shutil.copy2(root/'packaging/Info.plist',app/'Info.plist')
    Image.open(root/'source/gui/icons/sherloq_alpha.png').convert('RGBA').save(app/'Resources/SHERLOQ.icns',format='ICNS')
    subprocess.run(['codesign','--force','--sign','-',str(app.parent)],check=True)
    subprocess.run(['codesign','--verify','--strict',str(app.parent)],check=True)
    print(app.parent)


if __name__=='__main__':main()

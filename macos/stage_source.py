"""Export this fork into its native installation layout without changing code.

The repository preserves upstream's gui/ layout. The native integration expects
source/gui/ beside integration/, native/ and packaging/. This command bridges
those layouts. Installed runtimes and models are supplied by the Release.
"""
import argparse
import shutil
from pathlib import Path


def stage(destination):
    support = Path(__file__).resolve().parent
    repository = support.parent
    destination = destination.expanduser().resolve()
    if destination == repository or destination.is_relative_to(repository):
        raise ValueError('Choose a new destination outside the repository')
    destination.mkdir(parents=True, exist_ok=False)
    ignored = shutil.ignore_patterns('.git', '.DS_Store', '__pycache__', '*.pyc')
    for name in ('gui', 'docs', 'logo', 'screenshots', 'LICENSE', 'README.md', 'sherloq.py'):
        source = repository / name
        target = destination / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignored)
        else:
            shutil.copy2(source, target)
    for name in ('app_start.py', 'build', 'integration', 'native', 'packaging', 'requirements-macos-arm64.lock', 'tests'):
        source = support / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignored)
        else:
            shutil.copy2(source, target)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(stage(args.destination))

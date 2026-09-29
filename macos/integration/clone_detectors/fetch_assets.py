#!/usr/bin/env python3
"""Acquire pinned research assets without importing or installing upstream code.

Run from any directory with Python 3.11+. Existing verified assets are reused.
The manifest records local hashes, not an upstream authenticity guarantee.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'integration/clone_detectors'
DOWNLOADS = ROOT / 'downloads/clone_detectors'
SOURCES = ROOT / 'third_party/research/clone_detectors'
WEIGHTS = ROOT / 'models/external/clone_detectors'
MANIFEST = HERE / 'assets.json'
REPOS = {
 '01_d2prl': ('byc33/D2PRL', 'a4314b614ea3186b4fac98e9e96939e37f275fc5'),
 '02_forgeryscope': ('vlad3996/forgeryscope', '63101e28a12daea6f8c9f6b1cd1ff3b8ffd2d7c8'),
 '03_cmsegnet_reference': ('YoursEver/FakeParaEgg', 'ae09c70b19c5626787ea525a009039645a31205b'),
 '04_mgcfdn': ('tuhanglsWorld/MGCFDN', '2f0827b7c5675753c55318ca244bc2c712d6cd47'),
 '05_sift_g2nn_ransac': ('oshima-yoppi/luc_pub', '6f5b6790c2e03ea14faf5bc58799bc53da915c78'),
}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'SHERLOQ-assets'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)

def fetch(asset):
    asset = dict(asset)
    path = ROOT / asset['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            req = urllib.request.Request(asset['url'], headers={'User-Agent': 'Mozilla/5.0 SHERLOQ-assets'})
            with urllib.request.urlopen(req, timeout=60) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type and asset.get('kind') not in ('reference',):
                    raise ValueError('HTML response instead of expected binary asset')
                temporary = path.with_name(path.name + '.part')
                with temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream, 1024 * 1024)
            if asset.get('size_expected') and temporary.stat().st_size != asset['size_expected']:
                raise ValueError('Unexpected file size')
            temporary.replace(path)
        actual_hash = digest(path)
        if asset.get('sha256') and asset['sha256'] != actual_hash:
            raise ValueError('SHA256 differs from pinned manifest')
        asset.update(status='downloaded', size=path.stat().st_size, sha256=actual_hash)
        print('OK', asset['id'], asset['size'], flush=True)
    except Exception as exc:
        asset.update(status='unavailable', error=str(exc))
        print('UNAVAILABLE', asset['id'], str(exc), flush=True)
    return asset

def unpack_tar(path, destination):
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.extracting')
    temporary.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError('Unsupported archive member: ' + member.name)
            target = temporary.joinpath(*parts)
            if not target.resolve().is_relative_to(temporary.resolve()):
                raise ValueError('Unsafe archive member')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    temporary.rename(destination)

def unpack_zip(path, destination):
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.extracting')
    temporary.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = temporary / member.filename
            if not target.resolve().is_relative_to(temporary.resolve()) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe ZIP member')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    temporary.rename(destination)

def link(path, target):
    if not target.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    value = os.path.relpath(target, path.parent)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError('Conflicting symlink: ' + str(path))
    elif path.exists():
        raise ValueError('Refusing to replace existing path: ' + str(path))
    else:
        path.symlink_to(value, target_is_directory=target.is_dir())

def main():
    for path in (HERE, DOWNLOADS, SOURCES, WEIGHTS):
        path.mkdir(parents=True, exist_ok=True)
    assets = []
    for key, (repo, sha) in REPOS.items():
        assets.append(dict(id=key+'_source', kind='source_archive', repo=repo, commit=sha,
                           url=f'https://codeload.github.com/{repo}/tar.gz/{sha}',
                           path=str((DOWNLOADS/(key+'.tar.gz')).relative_to(ROOT))))
    release = get_json('https://api.github.com/repos/vlad3996/forgeryscope/releases/tags/models-v1')
    (DOWNLOADS/'forgeryscope-release.json').write_text(json.dumps(release, indent=2))
    for item in release['assets']:
        assets.append(dict(id='02_'+item['name'], kind='weights', url=item['browser_download_url'],
                           size_expected=item['size'], path=str((WEIGHTS/'02_forgeryscope'/item['name']).relative_to(ROOT))))
    for key, url in {
      '03_cmsegnet_code': 'https://www.dropbox.com/scl/fi/d70p1q7sh2qd0o983cbs5/FongYi_code.zip?rlkey=i3jdpdbhx40d69lwy3m06ov05&dl=1',
      '03_cmsegnet_weights': 'https://www.dropbox.com/scl/fi/aym6cdp84mqa6iehzf80h/prediction_folder_FongYi.zip?rlkey=74vzkqo2up7gg2wakemkropls&dl=1',
    }.items():
        assets.append(dict(id=key, kind='zip_archive', url=url, path=str((DOWNLOADS/(key+'.zip')).relative_to(ROOT))))
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old_assets = {a['id']: a for a in previous.get('assets', [])}
    for asset in assets:
        if old_assets.get(asset['id'], {}).get('sha256'):
            asset['sha256'] = old_assets[asset['id']]['sha256']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, assets))
    manifest = dict(schema_version=1, acquired_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), assets=results)
    # Preserve separately acquired assets.
    ids = {a['id'] for a in results}
    manifest['assets'].extend(a for a in previous.get('assets', []) if a['id'] not in ids)
    MANIFEST.write_text(json.dumps(manifest, indent=2)+'\n')
    for key in REPOS:
        if next(a for a in results if a['id']==key+'_source')['status']=='downloaded':
            unpack_tar(DOWNLOADS/(key+'.tar.gz'), SOURCES/key)
            link(HERE/'sources'/key, SOURCES/key)
    for key, target in [('03_cmsegnet_code', SOURCES/'03_cmsegnet'), ('03_cmsegnet_weights',WEIGHTS/'03_cmsegnet')]:
        if next(a for a in results if a['id']==key)['status']=='downloaded':
            unpack_zip(DOWNLOADS/(key+'.zip'), target)
            link(HERE/('sources' if key.endswith('code') else 'weights')/'03_cmsegnet',target)
    for key in REPOS:
        link(HERE/'weights'/key,WEIGHTS/key)
    mgc_zip = DOWNLOADS/'04_mgcfdn_weights.zip'
    if mgc_zip.exists():
        expected = old_assets.get('04_mgcfdn_weights', {}).get('sha256')
        if expected and digest(mgc_zip) != expected:
            raise ValueError('MGCFDN archive SHA256 mismatch')
        unpack_zip(mgc_zip, WEIGHTS/'04_mgcfdn')
        link(HERE/'weights/04_mgcfdn', WEIGHTS/'04_mgcfdn')
    link(HERE/'downloads',DOWNLOADS)
    for name,target in {
       'python': ROOT/'venv/bin/python',
       'lightglue_source':ROOT/'source/gui/sherloq_app/vendor/lightglue',
       'sift_lightglue.pth':ROOT/'models/external/sift_lightglue.pth',
       'aliked_lightglue.pth':ROOT/'models/external/aliked_lightglue.pth',
       'aliked-n16.pth':ROOT/'models/external/aliked-n16.pth',
    }.items():
        link(HERE/'existing'/name,target)
    print('Manifest:',MANIFEST,flush=True)

if __name__ == '__main__':
    main()

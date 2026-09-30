"""Synthetic download checks: no checkpoint or network required."""
from pathlib import Path
import hashlib,importlib.util,io,json,tempfile
from unittest.mock import patch
support=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('model_installer',support/'install_d2prl_model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=b'controlled model fixture';asset=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path='models/external/d2prl.pth',url='https://example.invalid/d2prl.pth')
cases=[]
for case in ('download','local-file','wrong-hash','oversized','interrupted','unknown-existing','symlink','concurrent-file'):
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory).resolve();(root/'source/gui').mkdir(parents=True);target=root/asset['path'];local=root/'input';local.write_bytes(data)
  target.parent.mkdir(parents=True)
  if case=='unknown-existing':target.write_bytes(b'keep this file')
  if case=='symlink':target.symlink_to(local)
  payload=b'x'*len(data) if case=='wrong-hash' else data+b'x' if case=='oversized' else data
  def open_source(*args,**kwargs):
   if case=='interrupted':raise OSError('injected network failure')
   return io.BytesIO(payload)
  original_link=module.os.link
  def concurrent_link(src,dst):
   if case=='concurrent-file':Path(dst).write_bytes(b'concurrent model')
   return original_link(src,dst)
  with patch.object(module.urllib.request,'urlopen',side_effect=open_source) as network,patch.object(module.os,'link',side_effect=concurrent_link):
   if case in ('download','local-file'):
    assert module.install(root,asset,local if case=='local-file' else None)==target
    assert module.install(root,asset)==target
    assert target.read_bytes()==data
    assert network.call_count==(1 if case=='download' else 0)
   else:
    try:module.install(root,asset)
    except (ValueError,OSError):pass
    else:raise AssertionError(case)
    if case=='unknown-existing':assert target.read_bytes()==b'keep this file'
    elif case=='symlink':assert target.is_symlink() and local.read_bytes()==data
    elif case=='concurrent-file':assert target.read_bytes()==b'concurrent model'
    else:assert not target.exists()
   assert not list(target.parent.glob('.d2prl-*.download'))
  cases.append(case)
print(json.dumps(dict(passed=True,cases=cases)))

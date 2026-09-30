import sys,json,hashlib,tempfile,io,zlib,threading,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source'))
from gui.sherloq_app.core.model_store import Store,Cancelled,feature_for
DATA=bytes(range(256))*12000
compressed=zlib.compress(DATA)[2:-4]
def asset(payload=DATA,encoding='identity',sha=None):return dict(path='models/model.bin',bytes=len(DATA),sha256=sha or hashlib.sha256(DATA).hexdigest(),encoding=encoding,parts=[dict(url='https://example.invalid/model',bytes=len(payload),offset=27,range=True)])
def catalog(a):return dict(schema=1,assets={'model':a},features={'model':{'assets':['model']},'twice':{'assets':['model'],'requires':['model']}})
class Response(io.BytesIO):
 def __init__(self,data,start,end):super().__init__(data);self.status=206;self.headers={'Content-Range':f'bytes {start}-{end}/{len(DATA)+27}'}
class Opener:
 def __init__(self,data=DATA):self.data=data;self.requests=[];self.wrong=False
 def __call__(self,request,timeout):
  start,end=map(int,request.headers['Range'][6:].split('-'));self.requests.append((start,end));return Response(self.data[start-27:end-26],start+int(self.wrong),end)
checks=[]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);op=Opener();store=Store(root,catalog(asset()),op);flag=[False]
 def progress(n,total,text):
  if n>=1048576:flag[0]=True
 try:store.ensure('model',lambda:flag[0],progress)
 except Cancelled:pass
 else:raise AssertionError('Cancellation ignored')
 assert not (root/'models/model.bin').exists()
 store.ensure('twice');assert op.requests[1][0]==27+1048576
 assert (root/'models/model.bin').read_bytes()==DATA
 n=len(op.requests);store.ensure('model');assert len(op.requests)==n
 checks.extend(['cancel','range-resume','local-reuse','dependency-deduplication'])
 (root/'models/model.bin').write_bytes(b'x'*len(DATA))
 try:store.ensure('model')
 except ValueError:pass
 else:raise AssertionError('Unknown existing file overwritten')
 checks.append('unknown-file-preserved')
for kind,payload in [('identity',DATA),('deflate',compressed)]:
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);op=Opener(payload);store=Store(root,catalog(asset(payload,kind)),op);store.ensure('model');assert (root/'models/model.bin').read_bytes()==DATA
  checks.append(kind+'-verified')
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);op=Opener();op.wrong=True;store=Store(root,catalog(asset()),op)
 try:store.ensure('model')
 except ValueError:pass
 else:raise AssertionError('Bad range accepted')
 assert not (root/'models/model.bin').exists();checks.append('wrong-range-refused')
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);store=Store(root,catalog(asset(sha='0'*64)),Opener())
 try:store.ensure('model')
 except ValueError:pass
 else:raise AssertionError('Bad hash accepted')
 assert not (root/'models/model.bin').exists() and not list(root.rglob('*.partial'));checks.append('bad-hash-no-install')
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);(root/'models').symlink_to('/tmp');
 try:Store(root,catalog(asset()),Opener())
 except ValueError:pass
 else:raise AssertionError('Symlink accepted')
 checks.append('symlink-refused')
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);op=Opener();a=Store(root,catalog(asset()),op);b=Store(root,catalog(asset()),op);errors=[]
 def run(s):
  try:s.ensure('model')
  except Exception as exc:errors.append(str(exc))
 threads=[threading.Thread(target=run,args=(s,)) for s in (a,b)]
 for t in threads:t.start()
 for t in threads:t.join()
 assert not errors and len(op.requests)==1;checks.append('concurrent-download-reused')
# A compressed ZIP record can cross two published release parts.
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);middle=len(compressed)//2;parts=[compressed[:middle],compressed[middle:]];a=asset(compressed,'deflate')
 a['parts']=[dict(url='https://example.invalid/'+str(i),bytes=len(b),offset=27,range=True) for i,b in enumerate(parts)]
 def fetch(req,timeout):
  i=int(req.full_url.rsplit('/',1)[1]);start,end=map(int,req.headers['Range'][6:].split('-'));return Response(parts[i][start-27:end-26],start,end)
 Store(root,catalog(a),fetch).ensure('model');assert (root/'models/model.bin').read_bytes()==DATA;checks.append('deflate-across-release-parts')
# A standalone server ignoring Range restarts safely instead of appending duplicates.
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);a=asset();a['parts']=[dict(url='https://example.invalid/file',bytes=len(DATA))]
 cache=root/'.model-downloads'/a['sha256'];cache.mkdir(parents=True);(cache/'0.partial').write_bytes(DATA[:512])
 def fetch(req,timeout):
  value=Response(DATA,0,len(DATA)-1);value.status=200;return value
 Store(root,catalog(a),fetch).ensure('model');assert (root/'models/model.bin').read_bytes()==DATA;checks.append('ignored-range-safe-restart')
assert feature_for('copy_move','PatchMatch Zernike')=='none'
assert feature_for('clone_detectors','Forgeryscope Auto')=='forgeryscope'
result=dict(passed=True,cases=checks);Path(__file__).with_name('contracts.json').write_text(json.dumps(result,indent=2));print(result)

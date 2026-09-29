"""Fetch the missing ViG graph layers from the authors' primary repository."""
import urllib.request,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];folder=ROOT/'third_party/research/clone_detectors/vig_dependency';folder.mkdir(exist_ok=True)
def get(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SHERLOQ-local-integration'}),timeout=30).read()
api='https://api.github.com/repos/huawei-noah/Efficient-AI-Backbones'
revision='f90e129b645c3b1684fe07cd361cd557d0ad71f7';entries=json.loads(get(api+'/contents/vig_pytorch/gcn_lib?ref='+revision));rows=[]
entries.append(dict(type='file',name='License.txt',download_url='https://raw.githubusercontent.com/huawei-noah/Efficient-AI-Backbones/'+revision+'/vig_pytorch/License.txt'))
for entry in entries:
 if entry['type']!='file':raise ValueError('Unexpected nested dependency directory')
 data=get(entry['download_url']);p=folder/entry['name'];p.write_bytes(data);rows.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),url=entry['download_url']))
(ROOT/'integration/clone_detectors/vig-dependency.json').write_text(json.dumps(dict(repository='https://github.com/huawei-noah/Efficient-AI-Backbones',commit=revision,files=rows),indent=2))
print(revision,len(rows))

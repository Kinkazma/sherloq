"""Build namespaced local adaptations; leave the inventoried upstream files intact."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;DEST=BASE/'sherloq_clone_models'
INDEX={r['path']:r['sha256'] for r in json.loads((BASE/'files.sha256.json').read_text())['files']}
INDEX.update({r['path']:r['sha256'] for r in json.loads((BASE/'vig-dependency.json').read_text())['files']})
records=[]
def make(source,target,replacements=(),definitions=False):
 p=ROOT/source;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==INDEX[source],source
 s=raw.decode('utf-8-sig')
 for a,b in replacements:
  assert a in s,(source,a);s=s.replace(a,b)
 if definitions:
  tree=ast.parse(s);lines=s.splitlines(keepends=True);s=''.join(''.join(lines[n.lineno-1:n.end_lineno])+'\n\n' for n in tree.body if (isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)) or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))) and not (isinstance(n,ast.ImportFrom) and n.module=='ptflops'))
 if source.endswith('.py'):s='# Local SHERLOQ adaptation; original: '+source+'\n# Source SHA256: '+INDEX[source]+'\n'+s
 path=DEST/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s)
 records.append(dict(source=source,source_sha256=INDEX[source],target=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),replacements=replacements,definitions_only=definitions))
DEST.mkdir(exist_ok=True);(DEST/'__init__.py').write_text('')
for package in ('cmsegnet','mgcfdn','mgcfdn/model','mgcfdn/backbones'):
 p=DEST/package;p.mkdir(parents=True,exist_ok=True);(p/'__init__.py').write_text('')
base='models/external/clone_detectors/03_cmsegnet/predict_folder/'
make(base+'model.py','cmsegnet/model.py',[('from Cor import Corr','from .Cor import Corr'),('else: self._init_weights()','else: pass  # Full checkpoint loaded strictly by the adapter.')],True)
make(base+'Cor.py','cmsegnet/Cor.py',[('key = str(x_in.shape) + str(rat_s)','key = (tuple(x_in.shape), rat_s, str(x_in.device), x_in.dtype)')],True)
base='third_party/research/clone_detectors/04_mgcfdn/'
for name in ('MGCFDN','EffNet_MGCFDN','MPDN_MGCFDN','TNT_MGCFDN','VIG_MGCFDN'):
 replacements=[]
 if name in ('MGCFDN','EffNet_MGCFDN'):replacements=[('weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1','weights=None')]
 else:replacements=[('from backbones.','from ..backbones.')]
 make(base+'model/'+name+'.py','mgcfdn/model/'+name+'.py',replacements,True)
make(base+'backbones/MPDN.py','mgcfdn/backbones/MPDN.py',definitions=True)
make(base+'backbones/TNT.py','mgcfdn/backbones/TNT.py',definitions=True)
make(base+'backbones/VIG.py','mgcfdn/backbones/VIG.py',[('from gcn_lib import','from ..gcn_lib import')],True)
for path in sorted((ROOT/'third_party/research/clone_detectors/vig_dependency').glob('*.py')):
 make(str(path.relative_to(ROOT)),'mgcfdn/gcn_lib/'+path.name,[('np.float)', 'float)')] if 'np.float)' in path.read_text() else [])
base='third_party/research/clone_detectors/02_forgeryscope/'
for path in sorted((ROOT/base/'forgeryscope').rglob('*.py')):
 relative=str(path.relative_to(ROOT/base));replacements=[]
 text=path.read_text()
 for a,b in [('from forgeryscope','from sherloq_clone_models.forgeryscope'),('from lightglue import LightGlue, SIFT, ALIKED, match_pair','from gui.sherloq_app.vendor.lightglue.lightglue import LightGlue\nfrom gui.sherloq_app.vendor.lightglue.sift import SIFT\nfrom gui.sherloq_app.vendor.lightglue.aliked import ALIKED\nfrom gui.sherloq_app.vendor.lightglue.utils import match_pair'),('from lightglue.utils import','from gui.sherloq_app.vendor.lightglue.utils import')]:
  if a in text:replacements.append((a,b))
 if relative.endswith('embedder/torch.py'):
  replacements += [("timm.data.resolve_model_data_config(hparams['model_name'])","timm.data.resolve_model_data_config(self.backbone)"),("torch.load(checkpoint_path, map_location='cpu')","torch.load(checkpoint_path, map_location='cpu', weights_only=True)")]
 if relative.endswith('matcher/lightglue.py'):
  replacements += [('self.matcher = LightGlue(features=', 'from gui.sherloq_app.core.forgeryscope_adapter import local_lightglue\n        self.matcher = local_lightglue(features='),('torch.cuda.synchronize()','if self.device.type == "mps": torch.mps.synchronize()'),('return None, None, None, f"Matching failed with error: {e}"','raise RuntimeError(f"Échec LightGlue : {e}") from e')]
 if relative.endswith('detector/yolo.py'):
  replacements += [('from ultralytics import YOLO','from gui.sherloq_app.core.forgeryscope_adapter import local_yolo as YOLO')]
 if relative.endswith('model_zoo.py'):
  replacements += [('with urlopen(url) as response, open(temp_destination, "wb") as handle:','raise FileNotFoundError("Poids absent : " + str(destination))\n    with urlopen(url) as response, open(temp_destination, "wb") as handle:')]
 make(base+relative,relative,replacements)
for source,target in [('third_party/research/clone_detectors/02_forgeryscope/LICENSE','forgeryscope/LICENSE'),('third_party/research/clone_detectors/04_mgcfdn/LICENSE','mgcfdn/LICENSE'),('third_party/research/clone_detectors/vig_dependency/License.txt','mgcfdn/gcn_lib/License.txt')]:
 make(source,target)
for path in [DEST/'__init__.py']+[DEST/package/'__init__.py' for package in ('cmsegnet','mgcfdn','mgcfdn/model','mgcfdn/backbones')]:
 records.append(dict(target=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),generated='empty namespace initializer'))
(BASE/'runtime-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print('Prepared',len(records),'verified local adaptations')

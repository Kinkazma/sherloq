"""Verify and relocate this installation. Run with system Python 3.11+ on macOS."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MAGIC={b'\xcf\xfa\xed\xfe',b'\xce\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'}
def run(args):return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True)
def main():
 if sys.platform!='darwin' or platform.machine()!='arm64':raise SystemExit('This installed runtime requires macOS ARM64.')
 manifest=json.loads((ROOT/'INSTALLATION-MANIFEST.json').read_text())
 for row in manifest['files']:
  p=ROOT/row['path']
  if 'symlink' in row:
   # ZIP extraction tools vary in symlink support: the declared link is authoritative.
   if not p.is_symlink():
    if p.exists():p.unlink()
    p.symlink_to(row['symlink'])
   assert os.readlink(p)==row['symlink'],row['path']
   assert p.resolve().is_relative_to(ROOT),row['path']
  else:
   with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
   assert actual==row['sha256'],'Checksum mismatch: '+row['path']
   p.chmod(row['mode'])
 relocation=json.loads((ROOT/'RELOCATION.json').read_text());old=relocation['neutral_root'];changed=[]
 # Restore text launch paths and metadata. Never replace variable-length paths
 # in native binary data; install_name_tool handles load commands instead.
 for edit in relocation['edits']:
  p=ROOT/edit['path']
  if edit['macho']:continue
  b=p.read_bytes()
  try:text=b.decode('utf-8')
  except UnicodeDecodeError:continue
  if old in text:p.write_text(text.replace(old,str(ROOT)));changed.append(edit['path'])
 for row in manifest['files']:
  if 'symlink' in row:continue
  p=ROOT/row['path']
  with p.open('rb') as f:magic=f.read(4)
  if magic not in MAGIC:continue
  lines=run(['otool','-L',str(p)]).splitlines()[1:];modified=False
  for line in lines:
   dep=line.strip().split(' (')[0]
   if dep.startswith(old+'/'):
    target=ROOT/dep[len(old)+1:]
    if target==p:run(['install_name_tool','-id','@rpath/'+p.name,str(p)])
    else:run(['install_name_tool','-change',dep,'@loader_path/'+os.path.relpath(target,p.parent),str(p)])
    modified=True
  # Neutralized Mach-O bytes invalidate the original signature even if only
  # a debug/source path changed.
  if modified or any(e['path']==row['path'] and e['macho'] for e in relocation['edits']):
   run(['codesign','--force','--sign','-',str(p)]);changed.append(row['path'])
 for name in ('logs','cache','native/runtime','build','tests','integration/clone_detectors/yolo-config'):(ROOT/name).mkdir(parents=True,exist_ok=True)
 run([str(ROOT/'venv/bin/python'),str(ROOT/'packaging/build_app.py')])
 result={'verified_files':len(manifest['files']),'relocated_files':len(changed),'bundle_built':True}
 (ROOT/'RESTORE-RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

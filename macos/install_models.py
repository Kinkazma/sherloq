"""Prepare an installed SHERLOQ copy for offline use; downloads data, not models into RAM."""
import argparse,json,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('installation',type=Path);p.add_argument('--feature');p.add_argument('--all',action='store_true');args=p.parse_args()
 root=args.installation.expanduser().resolve(strict=True)
 if not (root/'source/gui').is_dir():p.error('Choose an installed SHERLOQ folder')
 if bool(args.feature)==bool(args.all):p.error('Choose --feature NAME or --all')
 sys.path.insert(0,str(root/'source'))
 from gui.sherloq_app.core.model_store import Store
 store=Store(root)
 names=store.catalog['features'] if args.all else [args.feature]
 try:
  for name in names:
   print(name,flush=True)
   store.ensure(name,progress=lambda n,total,text:print(f'\r{text}: {n/1048576:.1f}/{total/1048576:.1f} MiB',end='',flush=True))
   print()
 except KeyboardInterrupt:p.exit(130,'\nCancelled; rerun to resume.\n')
 except (OSError,ValueError,KeyError) as exc:p.exit(1,str(exc)+'\n')
 print('Requested models verified; ready for offline use.')
if __name__=='__main__':main()

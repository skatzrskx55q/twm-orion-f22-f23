#!/usr/bin/env python3
"""Run native research entry points in an isolated copy. Python 3.13 recommended."""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keep',type=Path,help='New directory to preserve generated results and the copied research tree')
    p.add_argument('mode',choices=['check','demo','native'])
    p.add_argument('args',nargs=argparse.REMAINDER,help='Native Python arguments after native --')
    a=p.parse_args(); args=a.args[1:] if a.args[:1]==['--'] else a.args
    if a.mode!='native' and args:p.error('extra arguments only allowed in native mode')
    if a.mode=='native' and not args:p.error('native requires a Python script or -m module')
    if a.mode=='native' and args[0] in ('-c','-i'):p.error('use a native script or module')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    temp=None
    if a.keep:
        base=a.keep.resolve()
        if base.exists():p.error('--keep must name a new directory')
        if base.is_relative_to(ROOT):p.error('--keep must be outside the release directory')
        base.mkdir(parents=True)
    else:
        temp=tempfile.TemporaryDirectory(prefix='twm-research-');base=Path(temp.name)
    try:
        shutil.copytree(ROOT/'research',base/'research')
        cwd=base/CONFIG['native_directory'];env['PYTHONPATH']=str(cwd)
        cmd=[sys.executable,*(args if a.mode=='native' else CONFIG['commands'][a.mode])]
        print('Working directory:',cwd,flush=True);print('Command:',cmd,flush=True)
        return subprocess.run(cmd,cwd=cwd,env=env).returncode
    finally:
        if temp:temp.cleanup()
if __name__=='__main__':raise SystemExit(main())

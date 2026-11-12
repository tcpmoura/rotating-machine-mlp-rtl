"""Portable arithmetic/interface checks; all generated files stay in build/."""
from pathlib import Path
import argparse,json,shutil,subprocess,re,datetime
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--simulator',choices=['iverilog','xrun'],default='iverilog');p.add_argument('--vectors',type=Path);args=p.parse_args()
    vector=(args.vectors or ROOT/'tests/golden_vectors.txt').resolve()
    n=sum(bool(l.strip()) for l in vector.read_text().splitlines());assert 1<n<=28080
    exe=shutil.which(args.simulator)
    if not exe:raise SystemExit(f'{args.simulator} is not in PATH')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');dest=ROOT/'build'/stamp;dest.mkdir(parents=True)
    summary=[]
    for kind in ['folded','parallel']:
      for mode in [0,1]:
        folder=dest/f'{kind}_{mode}';folder.mkdir()
        shutil.copy2(vector,folder/'vectors.txt');shutil.copy2(ROOT/'model/weights_rom.mem',folder/'weights_rom.mem')
        files=[ROOT/'tests/tb_core_stream.sv',ROOT/'rtl'/f'core_{kind}.sv']
        if args.simulator=='iverilog':
            cmd=[exe,'-g2012','-I',str(ROOT/'rtl'),'-D',f'DUT=core_{kind}']
            if kind=='folded':cmd+=['-D','FOLDED']
            cmd+=['-s','tb_core_stream','-o','sim.out',*map(str,files)]
            subprocess.run(cmd,cwd=folder,check=True,capture_output=True,text=True)
            cmd=['vvp','sim.out',f'+MODE={mode}']
        else:
            cmd=[exe,'-64bit','-sv','-timescale','1ns/1ps','-incdir',str(ROOT/'rtl'),f'+define+DUT=core_{kind}']
            if kind=='folded':cmd+=['+define+FOLDED']
            cmd+=['-top','tb_core_stream',*map(str,files),f'+MODE={mode}','-l','xrun.log']
        proc=subprocess.run(cmd,cwd=folder,capture_output=True,text=True,timeout=1800)
        (folder/'console.log').write_text(proc.stdout+proc.stderr,encoding='utf-8');proc.check_returncode()
        match=re.search(r'STREAM_PASS vectors=(\d+) mode=(\d+) latency=(\d+)\.\.(\d+) interval=(\d+)\.\.(\d+) elapsed=(\d+) stalls=(\d+)',proc.stdout)
        assert match and int(match[1])==n and int(match[2])==mode
        lat,ii=(116,118) if kind=='folded' else (3,5)
        assert int(match[3])==int(match[4])==lat
        if mode==0:assert int(match[5])==int(match[6])==ii
        else:assert int(match[8])>0
        summary.append(dict(core=kind,mode=mode,result=match[0]));print(kind,match[0],flush=True)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'PASS. Generated files: {dest}')
if __name__=='__main__':main()

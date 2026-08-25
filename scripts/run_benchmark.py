from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trace_r.benchmark import write_outputs

if __name__=='__main__':
    out=write_outputs(ROOT)
    print('Generated benchmark artifacts:')
    for k,v in out.items(): print(f'{k}: {len(v)} rows')

"""Explicit Stage3 fixed-training entry point; never executes at import."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from terminal_marl.core.rotating_slots.stage3_learning.learning import train_model

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--input','--dataset',dest='dataset',required=True)
    p.add_argument('--output','--output-dir',dest='output',required=True)
    p.add_argument('--binding','--fingerprint',dest='binding',required=True)
    p.add_argument('--config')
    p.add_argument('--deadline',type=float)
    a=p.parse_args(argv)
    result=train_model(a.dataset,a.output,a.binding,config=a.config,deadline=a.deadline)
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

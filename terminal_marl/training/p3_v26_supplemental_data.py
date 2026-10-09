"""One declared supplemental train cycle; no Q or learning entry point."""
import argparse
import json
from pathlib import Path
from .p3_v26_course_data import collect_resource_case,validated_resource_case
from .p3_v3_bc_learnability import require


PROTOCOL='p3_v26_medium_dense_supplemental_train_20261009_v1'
TRAIN_CASES=((893604,'medium12_mixed'),(893605,'dense24_fixed'),(893606,'dense24_mixed'))
SMOKE_SEED=892872


def collect(output,index,*,smoke=False):
    require(type(index) is int and 1<=index<=(1 if smoke else len(TRAIN_CASES)),
            'undeclared supplemental train index')
    seed,scenario=(SMOKE_SEED,'medium12_mixed') if smoke else TRAIN_CASES[index-1]
    return collect_resource_case(output,index=index,seed=seed,scenario=scenario,smoke=smoke,protocol_name=PROTOCOL)


def validated_case(directory,index):
    require(type(index) is int and 1<=index<=len(TRAIN_CASES),'undeclared supplemental train index')
    seed,scenario=TRAIN_CASES[index-1]
    return validated_resource_case(directory,seed=seed,scenario=scenario,protocol_name=PROTOCOL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    result=collect(args.output,args.index,smoke=args.smoke)
    print(json.dumps(dict(status=result['status'],seed=result['seed'],error=result['error'])))
    return 0 if result['status'] in ('SMOKE_OK','COLLECTION_OK') else 2


if __name__=='__main__':raise SystemExit(main())

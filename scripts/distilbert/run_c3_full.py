"""Khóa kế hoạch C3 trước khi train; chạy ba seed tuần tự, lưu stdout/stderr thật."""
import os
import subprocess
import sys
from datetime import datetime, timezone
from src.models.distilbert_study import ROOT, read_json, write_json
from src.datasets.goemotions import sha256, REVISION


def main():
    config_path = ROOT/'configs/distilbert_full.json'
    config = read_json(config_path)
    folder = ROOT/'reports/c3_distilbert/full'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {
        'scope':'C3 owned by Huy; validation study only; not team final-test approval',
        'created_at_utc':datetime.now(timezone.utc).isoformat(), 'seeds':[42,123,2026],
        'config':config, 'config_sha256':sha256(config_path), 'data_revision':REVISION,
        'checkpoint_selection':'highest validation macro_f1 @0.5; earliest epoch if tied',
        'main_summary':'all 3 seeds, mean and sample std ddof=1, all 28 labels',
        'c3_representative_seed':'highest validation macro_f1 @0.5; smaller seed if tied; not winner among C1/C2/C3',
        'enhancement':'per-label threshold tuning separately for each of 3 selected checkpoints; validation only',
        'threshold_grid':[round(i*.05,2) for i in range(1,20)],
        'threshold_tie_rule':'nearest 0.5; then larger threshold',
        'rare_labels_train_defined':['grief','pride','relief','nervousness','embarrassment'],
        'test_used':False,
        'source_sha256':{p:sha256(ROOT/p) for p in ['src/models/distilbert_study.py','src/datasets/goemotions.py','src/evaluation/metrics.py','scripts/distilbert/train_distilbert.py']}}
    path = folder/'protocol.json'
    if path.exists():
        existing=read_json(path)
        for key in protocol:
            if key!='created_at_utc' and protocol[key]!=existing[key]:
                raise ValueError(f'Protocol changed: {key}. Do not mix experiments.')
    else:
        write_json(path,protocol)
    env=dict(os.environ, PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    for seed in protocol['seeds']:
        run=ROOT/f'data/processed/c3_distilbert/full/seed_{seed}'
        if (run/'run.json').exists():
            meta=read_json(run/'run.json')
            expected=dict(config,seed=seed)
            if meta['status']!='complete' or meta['config']!=expected or meta['source_sha256']!=protocol['source_sha256']:
                raise ValueError('Run có sẵn không khớp protocol')
            for name,digest in meta['artifact_sha256'].items():
                if sha256(run/name)!=digest:
                    raise ValueError(f'Sai hash {run/name}')
            print(f'Seed {seed}: complete artifact already verified',flush=True)
            continue
        log=folder/f'train_seed_{seed}.log'
        if log.exists():
            raise FileExistsError(f'Không ghi đè log cũ: {log}')
        print(f'START seed {seed}; real training log: {log}',flush=True)
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u','-m','scripts.distilbert.train_distilbert','--config',str(config_path),
                            '--seed',str(seed)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'COMPLETE seed {seed}',flush=True)
    print('PASS: all three C3 full seeds complete; no test evaluation.',flush=True)


if __name__=='__main__':
    # Chỉ giữ hệ thống thức trong lúc chạy; không sửa power plan, cho phép màn hình tắt.
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

"""Sau khi runner thật hoàn tất, kiểm/xuất bảng và thực thi notebook; không train thêm."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
from src.models.distilbert_study import ROOT, read_json, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runner-pid',type=int,required=True)
    parser.add_argument('--runner-log',type=Path,required=True)
    args=parser.parse_args()
    while psutil.pid_exists(args.runner_pid):
        time.sleep(5)
    if 'PASS: all three C3 full seeds complete' not in args.runner_log.read_text(encoding='utf-8'):
        raise RuntimeError('Runner chưa xác nhận đủ ba seed. Không xuất kết quả một phần.')
    output=ROOT/'reports/c3_distilbert/full'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    stages=[['scripts.distilbert.analyze_c3_full'],['scripts.distilbert.report_c3_full']]
    for stage in stages:
        print('START',stage[0],flush=True)
        subprocess.run([sys.executable,'-u','-m',*stage],cwd=ROOT,env=env,check=True)
    selected=read_json(output/'representative_c3.json')
    subprocess.run([sys.executable,'-u','-m','scripts.distilbert.verify_distilbert','--run',selected['run'],
        '--output',str(output/'verification.json')],cwd=ROOT,env=env,check=True)
    for module in ['scripts.distilbert.build_c3_full_notebook','scripts.distilbert.run_distilbert_notebook']:
        print('START',module,flush=True)
        subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT,env=env,check=True)
    write_json(output/'automatic_checks_complete.json',{
        'status':'passed','stages':['full_three_seeds','analysis','report','app_cli_checks','executed_notebook'],
        'remaining_manual_review':['error interpretation','plot inspection','demo restart evidence','documentation status'],
        'test_used':False})
    print('PASS: tables, report, demo checks and executed notebook ready for final review.',flush=True)


if __name__=='__main__':
    main()

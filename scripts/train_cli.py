"""VS Code/终端统一训练入口，支持7:2:1划分和断点续训。"""
import argparse
import copy
from datetime import datetime
from pathlib import Path
from nlms_dfe.common import read_config,project_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/train_window.json');p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--resume');p.add_argument('--output');p.add_argument('--control-file')
    args=p.parse_args()
    if args.control_file:
        import os
        os.environ['NLMS_CONTROL_FILE']=str(project_path(args.control_file))
    if args.resume:
        import torch
        c=copy.deepcopy(torch.load(project_path(args.resume),map_location='cpu',weights_only=True)['config'])
        c['training']['resume']=str(project_path(args.resume))
    else:c=read_config(args.config)
    if args.epochs is not None:c['training']['epochs']=args.epochs
    if args.batch_size is not None:c['training']['batch_size']=args.batch_size
    c['training']['output']=str(project_path(args.output or ('outputs/cli_training/'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
    from scripts.workbench_job import run
    job={'action':'train','config':c,'publish_model':False}
    if not args.resume:job['split_ratio']=[.7,.2,.1]
    print(run(job)['message'])

if __name__=='__main__':main()

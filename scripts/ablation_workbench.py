"""可移植的多模型实验队列。各任务独立进程，统一数据划分。"""
import copy
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from nlms_dfe.common import ROOT, write_json, read_config, project_path
from scripts.ablation_variants import VARIANTS


def prepare(job):
    from nlms_dfe.data.io import load_manifest
    config=job['config']; out=Path(job['output'])
    rows=load_manifest(project_path(config['data']['manifest']))
    groups=sorted({r['channel_source_id'] for r in rows})
    random.Random(job['split_seed']).shuffle(groups)
    if job['mode']=='holdout':
        from nlms_dfe.data.splitting import split_rows
        rows=split_rows(rows,job['split_seed'])
    else:
        for r in rows:r['split']='train'
    out.mkdir(parents=True,exist_ok=False)
    manifest=out/'index.csv'
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    load_manifest(manifest) # 再检查同一文件与来源组不跨集合。
    write_json(out/'experiment.json',job)
    tasks=[]
    for name in job['variants']:
        for seed in job['seeds']:
            c=copy.deepcopy(config)
            for path,value in VARIANTS[name]['changes'].items():
                target=c; parts=path.split('.')
                for part in parts[:-1]:target=target[part]
                target[parts[-1]]=value
            folder=out/f'{name}_seed{seed}'
            c['data']['manifest']=str(manifest)
            c['training'].update(seed=seed,mode=job['mode'],output=str(folder),patience=c['training']['epochs']+1)
            tasks.append({'id':folder.name,'variant':name,'label':VARIANTS[name]['label'],'seed':seed,'config':c,'status':'等待'})
    write_json(out/'tasks.json',tasks)
    return tasks


def child(path):
    from nlms_dfe.training.engine import train,evaluate
    from scripts.plot_training import plot
    task=read_config(path); c=task['config']; folder=Path(c['training']['output'])
    checkpoint=train(c)
    metrics=evaluate(checkpoint,folder/'evaluation') if c['training']['mode']=='holdout' else read_config(folder/'training_fit_metrics.json')
    plot(folder)
    write_json(folder/'result.json',dict(checkpoint=str(checkpoint),metrics=metrics,scope='组外测试' if c['training']['mode']=='holdout' else '训练拟合'))


def run(job):
    tasks=prepare(job); out=Path(job['output'])
    def execute(task):
        spec=out/(task['id']+'.json');write_json(spec,task)
        status=out/(task['id']+'.status.json')
        write_json(status,{'status':'运行中'})
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',NLMS_PROGRESS_FILE=str(out/(task['id']+'.progress.json')))
        try:
            with (out/(task['id']+'.log')).open('w',encoding='utf-8') as log:
                p=subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.ablation_workbench','--child',str(spec)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            write_json(status,{'status':'完成' if p.returncode==0 else '失败','exit_code':p.returncode})
        except Exception as e:write_json(status,{'status':'失败','error':str(e)})
    with ThreadPoolExecutor(max_workers=job['concurrency']) as pool:list(pool.map(execute,tasks))
    records=[]
    for task in tasks:
        path=Path(task['config']['training']['output'])/'result.json'
        if path.exists():
            r=read_config(path);m=r['metrics']
            records.append(dict(model=task['label'],seed=task['seed'],scope=r['scope'],ber=m['selected_ber_macro'],fixed_ber=m['fixed_ber_macro'],regret=m['mean_regret'],top1=m['top1_tie_aware_accuracy'],checkpoint=r['checkpoint']))
    if records:
        with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return {'action':'ablation','output':str(out),'message':f'多模型实验结束：成功{len(records)}/{len(tasks)}。请在消融页查看结果；失败任务有独立日志。'}


if __name__=='__main__':
    child(sys.argv[2])

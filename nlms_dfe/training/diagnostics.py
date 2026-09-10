"""独立诊断：固定目标、固定训练监测样本，不读取测试集。"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def measures(scores, ber, fixed, config):
    s=torch.as_tensor(scores,dtype=torch.float32);b=torch.as_tensor(ber,dtype=torch.float32)
    scale=(torch.quantile(b,.9,dim=1)-torch.quantile(b,.1,dim=1)).clamp_min(config['scale_floor'])
    target=(b-b.min(1,keepdim=True).values)/scale[:,None]
    lp=F.log_softmax(-s/config['temperature'],dim=1);q=F.softmax(-target/config['temperature'],dim=1)
    # 固定局部生成器，避免诊断改变训练随机序列。
    g=torch.Generator().manual_seed(12345)
    a=torch.randint(s.shape[1],(config['rank_pairs'],),generator=g);c=torch.randint(s.shape[1],a.shape,generator=g)
    gap=target[:,c]-target[:,a];w=gap.abs().clamp(max=1)
    rank=(w*F.softplus(-gap.sign()*(s[:,c]-s[:,a]))).sum()/w.sum().clamp_min(1)
    ties=b==b.min(1,keepdim=True).values
    raw={'regression':float(F.smooth_l1_loss(s,target)), 'soft':float(-(q*lp).sum(1).mean()),
         'regret_loss':float((lp.exp()*target).sum(1).mean()),'rank':float(rank),
         'classification':float(-((ties/ties.sum(1,keepdim=True))*F.log_softmax(-s,dim=1)).sum(1).mean())}
    selected=np.asarray(ber)[np.arange(len(ber)),np.asarray(scores).argmin(1)]
    delta=selected-np.asarray(ber)[:,fixed];regret=selected-np.asarray(ber).min(1)
    raw.update(ber=float(selected.mean()),fixed=float(np.asarray(ber)[:,fixed].mean()),regret=float(regret.mean()),
               p90_regret=float(np.quantile(regret,.9)),better=float((delta<0).mean()),worse=float((delta>0).mean()),
               entropy=float(-(lp.exp()*lp).sum(1).mean()))
    for k in (1,3,5):
        order=np.argsort(scores,axis=1)[:,:k]
        raw['top'+str(k)]=float((np.take_along_axis(ber,order,axis=1).min(1)<=np.asarray(ber).min(1)+1e-8).mean())
    return raw


def record(output, epoch, training, validation, optimization):
    path=Path(output)/'diagnostic_history.csv'
    row={'epoch':epoch,**{'train_'+k:v for k,v in training.items()},
         **{'val_'+k:v for k,v in validation.items()},**optimization}
    exists=path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)
    with path.open(encoding='utf-8') as f:history=list(csv.DictReader(f))
    message='记录不足：需要连续多轮观察；提示不是统计检验。'
    if validation and len(history)>=4:
        old=history[-4]
        if training['ber']<float(old['train_ber']) and validation['ber']>float(old['val_ber']):
            message='可能过拟合：训练监测BER比三轮前降低，但验证BER升高。请查看完整趋势及来源组差异。'
        else:message='最近三轮未触发训练改善、验证恶化的组合提示；不能据此证明没有过拟合。'
    elif not validation:message='全量拟合：没有验证集，无法据此判断泛化或过拟合。'
    (Path(output)/'diagnostic_status.json').write_text(json.dumps({'epoch':epoch,'message':message},ensure_ascii=False),encoding='utf-8')

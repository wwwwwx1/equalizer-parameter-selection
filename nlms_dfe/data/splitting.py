"""按来源组生成独立7:2:1索引，不修改用户原始索引。"""
import csv
import random
from collections import Counter
from pathlib import Path
from .io import load_manifest
from ..common import write_json


def split_rows(rows,seed):
    rows=[dict(row) for row in rows]
    groups=sorted({r['channel_source_id'] for r in rows})
    if len(groups)<3:raise ValueError('7:2:1来源组划分至少需要3个独立来源组。')
    random.Random(seed).shuffle(groups)
    test_n=max(1,round(len(groups)*.1));val_n=max(1,round(len(groups)*.2))
    test=set(groups[:test_n]);val=set(groups[test_n:test_n+val_n])
    for row in rows:row['split']='test' if row['channel_source_id'] in test else 'val' if row['channel_source_id'] in val else 'train'
    return rows


def write_split(source,folder,seed):
    # 重新划分前不依赖旧split字段；写出后严格检查文件和来源不跨集合。
    rows=split_rows(load_manifest(Path(source),require_split=False),seed)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    path=folder/'index.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    load_manifest(path)
    counts=dict(Counter(r['split'] for r in rows))
    summary={'seed':seed,'target_ratio':[.7,.2,.1],'group_counts':{s:len({r['channel_source_id'] for r in rows if r['split']==s}) for s in ('train','val','test')},'sample_counts':counts,'note':'按来源组约7:2:1；组大小不同时样本比例不一定精确。'}
    write_json(folder/'split_summary.json',summary)
    print(f"来源组7:2:1划分完成：训练{counts['train']}、验证{counts['val']}、测试{counts['test']}个样本。",flush=True)
    return path

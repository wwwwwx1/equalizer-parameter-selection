"""外部信道离线选参评估：冻结模型及固定基线，严格匹配参数元组。"""
import argparse
import copy
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from nlms_dfe.common import write_json
from nlms_dfe.data.io import read_field
from nlms_dfe.data.transforms import make_features
from nlms_dfe.data.dataset import collate_channels
from nlms_dfe.training.engine import load_checkpoint
from scripts.prepare_data import matlab_text


def identity(name):
    """不依赖盘符；比对文件身份及可识别的来源起点，不能证明内容独立。"""
    name = name.replace('\\', '/').split('/')[-1]
    match = re.search(r'start\.(\d+).*?index\.(\d+).*?dataID\.(-?\d+)', name)
    return '|'.join(match.groups()) if match else name


def parameter_order(reference, table):
    order = []
    for row in reference:
        matches = np.flatnonzero(np.isclose(table[:, 0], row[0], rtol=1e-6, atol=1e-8)
                                 & (table[:, 1] == row[1]) & (table[:, 2] == row[2]))
        if len(matches) != 1:
            raise ValueError(f'参数组合无法唯一匹配：{row}')
        order.append(int(matches[0]))
    if len(set(order)) != len(table) or len(reference) != len(table):
        raise ValueError('模型与结果候选集合不同。')
    return np.array(order)


def fixed_index(parameters, requested):
    values = np.asarray(requested, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('固定参数必须为三个有限数值：mu、N1、N2。')
    hits = np.flatnonzero(np.isclose(parameters[:,0], values[0], rtol=1e-6, atol=1e-8)
                         & (parameters[:,1] == values[1]) & (parameters[:,2] == values[2]))
    if len(hits) != 1:
        raise ValueError('固定组合不在294组候选表中，无法查询其BER。请重新选择。')
    return int(hits[0])


def plot_results(folder, requested=None):
    """只读取已完成结果；可用缓存BER更换固定基线，不重新预测。"""
    folder=Path(folder)
    metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows) != metrics['samples'] or not rows:
        raise ValueError('结果不完整，不能绘制全量结果图。')
    a=lambda k:np.array([float(r[k]) for r in rows])
    selected=a('selected_ber');baseline=a('fixed_ber');parameters=metrics['fixed_parameters']
    if requested is not None:
        with np.load(folder/'scores_and_ber.npz',allow_pickle=False) as archive:
            index=fixed_index(archive['candidates'],requested)
            if len(archive['ber'])!=len(rows):raise ValueError('BER缓存样本数不一致。')
            baseline=archive['ber'][:,index].copy();parameters=archive['candidates'][index].tolist()
    output=folder/'plots'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta=selected-baseline
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle(f'Fixed: mu={parameters[0]:g}, N1={parameters[1]:g}, N2={parameters[2]:g}')
    for values,label in [(selected,'Model'),(baseline,'Fixed'),(a('grid_best_ber'),'Grid minimum')]:axes[0,0].plot(values*100,label=label)
    axes[0,0].legend();axes[0,0].set(xlabel='Channel index',ylabel='BER (%)')
    axes[0,1].bar(['Model','Fixed','Grid minimum'],[selected.mean()*100,baseline.mean()*100,a('grid_best_ber').mean()*100]);axes[0,1].set(ylabel='Mean BER (%)')
    axes[1,0].hist(delta*100,bins=20);axes[1,0].axvline(0,color='black');axes[1,0].set(xlabel='Model - fixed (percentage points)',ylabel='Channels')
    axes[1,1].scatter(a('snr_db'),a('regret')*100);axes[1,1].set(xlabel='Recorded SNR (dB)',ylabel='Regret (percentage points)')
    for ext in ('png','pdf','svg'):fig.savefig(output/('comparison.'+ext),dpi=160)
    plt.close(fig)
    write_json(output/'comparison_metrics.json',dict(source=str(folder.resolve()),fixed_parameters=parameters,samples=len(rows),
        mean_grid_best_ber=metrics['mean_grid_best_ber'],mean_regret=metrics['mean_regret'],p90_regret=metrics['p90_regret'],
        top1=metrics['top1'],top3=metrics['top3'],top5=metrics['top5'],
        baseline_source='user_specified' if requested is not None else metrics.get('fixed_source','checkpoint'),
        mean_selected_ber=float(selected.mean()),mean_fixed_ber=float(baseline.mean()),
        relative_ber_reduction=float(1-selected.mean()/baseline.mean()) if baseline.mean() else None,
        better_fraction=float((delta<0).mean()),worse_fraction=float((delta>0).mean()),tie_fraction=float((delta==0).mean())))
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['channel_name','selected_ber','fixed_ber','delta_fixed'])
        writer.writerows((r['channel_name'],float(s),float(b),float(d)) for r,s,b,d in zip(rows,selected,baseline,delta))
    print('绘图完成：'+str(output),flush=True)
    return output


def run(job):
    torch.set_num_threads(4)
    checkpoint = Path(job['checkpoint']).resolve()
    device = 'cuda' if job.get('device', 'auto') == 'auto' and torch.cuda.is_available() else job.get('device', 'cpu')
    if device == 'auto': device = 'cpu'
    model, saved = load_checkpoint(checkpoint, device)
    if saved['config']['training'].get('target_metric') != 'MeanCodedBER':
        raise ValueError('只接受MeanCodedBER模型。')
    out = Path(job['output']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'job.json', job)
    config = copy.deepcopy(saved['config']['data'])
    config['crop'].update(already_cropped=job.get('already_cropped', False),
                          main_path_matlab=int(job.get('main_path', 626)),allow_short=job.get('allow_short',False))
    # 时间范围、池化和特征保持checkpoint定义，避免测试时偷偷改变模型输入。
    write_json(out/'effective_data_config.json', config)
    reference = saved['candidates'].cpu().numpy()
    candidate_ids = saved['candidate_ids'].cpu().numpy()
    fixed = int(saved['fixed_index'])
    if job.get('fixed_parameters') is not None:
        fixed=fixed_index(reference,job['fixed_parameters'])
    fixed_source='user_specified' if job.get('fixed_parameters') is not None else 'checkpoint'
    snapshot = saved.get('manifest_snapshot', [])
    seen = {identity(r['channel_path']): r['split'] for r in snapshot}
    channels = sorted(Path(job['channels']).rglob('*.mat'))
    if not channels: raise ValueError('没有找到信道MAT。')
    # 先索引结果；多结果同名不静默选择，避免SNR配错。
    result_map = {}
    for path in sorted(Path(job['results']).rglob('grid_*.mat')):
        name = matlab_text(read_field(path, 'currName')).replace('\\', '/').split('/')[-1]
        result_map.setdefault(name, []).append(path)
    print(f'模型第{saved.get("epoch")}轮；设备{device}；待测试{len(channels)}个信道', flush=True)
    records, score_rows, ber_rows = [], [], []
    csv_path = out/'predictions.csv'
    for number, path in enumerate(channels, 1):
        matches = result_map.get(path.name, [])
        if len(matches) != 1: raise ValueError(f'{path.name}对应结果数量{len(matches)}，要求唯一。')
        result = matches[0]
        snr = float(read_field(result, 'currSNR').item())
        table = np.column_stack([read_field(result, key).ravel() for key in ('param_delt','param_N1','param_N2')])
        order = parameter_order(reference, table)
        mc = read_field(result, 'coded_ber_after_mc')
        if mc.ndim != 2 or mc.shape[0] != len(table) or not np.isfinite(mc).all() or np.any((mc<0)|(mc>1)):
            raise ValueError(f'无效coded BER矩阵：{result}')
        raw_ber = mc.mean(axis=1)
        sidecar = result.with_suffix('.csv')
        if sidecar.exists():
            with sidecar.open(encoding='utf-8-sig', newline='') as f:
                rows = sorted(csv.DictReader(f), key=lambda r:int(r['ParamIndex']))
            np.testing.assert_allclose([float(r['MeanCodedBER']) for r in rows],raw_ber,atol=1e-12,rtol=1e-7)
        ber = raw_ber[order]
        if not np.isfinite(snr): raise ValueError('SNR非有限值')
        h = read_field(path, config['fields']['channel'], config.get('hdf5_matlab_order',True))
        x, power = make_features(h, config)
        batch = collate_channels([dict(x=x,scalars=torch.tensor([snr,power],dtype=torch.float32))])
        with torch.inference_mode():
            scores = model(batch['x'].to(device),batch['scalars'].to(device),batch['padding_mask'].to(device))[0].cpu().numpy()
        if not np.isfinite(scores).all(): raise ValueError('模型评分异常')
        ranking = np.argsort(scores, kind='stable')
        selected = int(ranking[0])
        record = dict(channel_name=path.name,snr_db=snr,candidate_id=int(candidate_ids[selected]),
                      mu=float(reference[selected,0]),N1=int(reference[selected,1]),N2=int(reference[selected,2]),
                      selected_ber=float(ber[selected]),fixed_ber=float(ber[fixed]),grid_best_ber=float(ber.min()),
                      regret=float(ber[selected]-ber.min()),delta_fixed=float(ber[selected]-ber[fixed]),
                      known_split=seen.get(identity(path.name),'unmatched_identity'),
                      source_result=str(result),channel_path=str(path),raw_shape=str(h.shape),short_record=bool(h.shape[0]<config['crop']['time_stop_matlab']))
        for k in (1,3,5): record[f'top{k}_hit']=bool(ber[ranking[:k]].min() <= ber.min()+saved['config']['evaluation'].get('tie_tolerance',1e-8))
        records.append(record);score_rows.append(scores);ber_rows.append(ber)
        with csv_path.open('a',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(record))
            if number==1: writer.writeheader()
            writer.writerow(record)
        print(f'{number}/{len(channels)} BER={record["selected_ber"]:.6f} fixed={record["fixed_ber"]:.6f} {path.name}',flush=True)
    a = lambda key: np.array([r[key] for r in records])
    delta = a('delta_fixed')
    metrics = dict(samples=len(records),checkpoint=str(checkpoint),epoch=int(saved['epoch']),device=device,
                   mean_selected_ber=float(a('selected_ber').mean()),mean_fixed_ber=float(a('fixed_ber').mean()),
                   mean_grid_best_ber=float(a('grid_best_ber').mean()),mean_regret=float(a('regret').mean()),
                   p90_regret=float(np.quantile(a('regret'),.9)),better_fraction=float((delta<0).mean()),
                   worse_fraction=float((delta>0).mean()),tie_fraction=float((delta==0).mean()),
                   fixed_parameters=reference[fixed].tolist(),fixed_source=fixed_source,known_identity_overlap=sum(r['known_split']!='unmatched_identity' for r in records),
                   scope='外部数据离线BER表回放；非重新MATLAB仿真；未匹配身份不证明来源独立')
    metrics['relative_ber_reduction']=float(1-metrics['mean_selected_ber']/metrics['mean_fixed_ber']) if metrics['mean_fixed_ber'] else None
    for k in (1,3,5): metrics[f'top{k}']=float(a(f'top{k}_hit').mean())
    write_json(out/'metrics.json',metrics)
    # 独立测试图使用实际逐信道结果，不补造训练历史。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key in ('selected_ber','fixed_ber','grid_best_ber'): axes[0,0].plot(a(key)*100,label=key)
    axes[0,0].set(title='Per-channel MeanCodedBER',xlabel='Channel index',ylabel='BER (%)');axes[0,0].legend()
    axes[0,1].bar(['Model','Fixed','Grid minimum'],[a(k).mean()*100 for k in ('selected_ber','fixed_ber','grid_best_ber')]);axes[0,1].set(ylabel='Mean BER (%)')
    fig.suptitle(f'Fixed: mu={reference[fixed,0]:g}, N1={reference[fixed,1]:g}, N2={reference[fixed,2]:g}')
    axes[1,0].hist(delta*100,bins=20);axes[1,0].axvline(0,color='black');axes[1,0].set(xlabel='Model - fixed (percentage points)',ylabel='Channels')
    axes[1,1].scatter(a('snr_db'),a('regret')*100);axes[1,1].set(xlabel='Recorded SNR (dB)',ylabel='Regret (percentage points)')
    for ext in ('png','svg','pdf'):fig.savefig(out/('comparison.'+ext),dpi=160)
    plt.close(fig)
    np.savez_compressed(out/'scores_and_ber.npz',scores=score_rows,ber=ber_rows,candidates=reference,candidate_ids=candidate_ids)
    savemat(out/'batch_predictions.mat',dict(channel_names=a('channel_name').astype(object),snr_db=a('snr_db'),
            mu=a('mu'),N1=a('N1'),N2=a('N2'),candidate_id=a('candidate_id'),selected_ber=a('selected_ber'),
            fixed_ber=a('fixed_ber'),grid_best_ber=a('grid_best_ber')),do_compression=True)
    report = ['# 外部信道模型测试结果','',metrics['scope'],'',f'模型：{checkpoint}，第{saved["epoch"]}轮。',
              f'共{len(records)}个信道；固定参数{reference[fixed].tolist()}；来源：{fixed_source}（checkpoint为训练基线，user_specified为用户指定）。','',
              '| 指标 | 值 |','|---|---|']
    report += [f'| {key} | {value} |' for key,value in metrics.items() if isinstance(value,(int,float))]
    report += ['', '身份检查只按文件名中的start/index/dataID匹配，不能证明与训练信道物理来源独立。',
               '测试输入采用结果文件currSNR，标签使用coded_ber_after_mc的重复均值并核对MeanCodedBER CSV。',
               '候选表按mu/N1/N2匹配，不假定候选编号与顺序相同。',
               f'短记录数量：{sum(r["short_record"] for r in records)}；允许短记录时，时间窗口取到实际末尾，不补零。',
               '模型训练窗口与历史均衡器实际使用窗口可能不同；本次是预测查表评估，不证明仿真协议完全一致。',
               '后续反复根据这批信道调参，会使其成为开发验证集；最终结论仍需新的独立测试数据。','', '![结果对比](comparison.png)']
    (out/'测试报告.md').write_text('\n'.join(report),encoding='utf-8')
    print('完成：'+str(out),flush=True)
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    job=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if job.get('action')=='plot':plot_results(job['folder'],job.get('fixed_parameters'))
    else:run(job)

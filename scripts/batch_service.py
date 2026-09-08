"""为每个信道随机指定SNR，只做参数预测，汇总为MATLAB可直接读取的文件。"""
import csv
import hashlib
import json
import secrets
import sys
from datetime import datetime
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from scipy.io import savemat, loadmat
from nlms_dfe.common import active_model, project_path, write_json
from nlms_dfe.data.io import read_field
from nlms_dfe.predict import Predictor


def save_mat(path, records, metadata):
    """results是结构体数组；另外提供同序列数组，便于直接写MATLAB循环。"""
    fields = ['channel_name','channel_path','snr_db','mu','N1','N2','candidate_id','score','success','error']
    results = np.empty((len(records),1), dtype=[(key,'O') for key in fields])
    for i,row in enumerate(records):
        for key in fields:
            results[key][i,0] = row[key]
    column = lambda key: np.array([r[key] for r in records],dtype=np.float64)[:,None]
    savemat(path, {
        'results':results,
        'channel_names':np.array([r['channel_name'] for r in records],dtype=object)[:,None],
        'channel_paths':np.array([r['channel_path'] for r in records],dtype=object)[:,None],
        'snr_db':column('snr_db'), 'mu':column('mu'), 'N1':column('N1'), 'N2':column('N2'),
        'parameters':np.column_stack([column(k).ravel() for k in ('mu','N1','N2')]),
        'candidate_id':column('candidate_id'),
        'success':np.array([r['success'] for r in records],dtype=np.uint8)[:,None],
        'random_seed':metadata['random_seed'],
        'snr_source':'randomly_assigned_for_parameter_validation_not_measured',
        'model_path':metadata['model_path'],
        'target_metric':metadata['target_metric'],
        'metadata_json':json.dumps(metadata,ensure_ascii=False)
    }, do_compression=True, long_field_names=True)


def run_batch(config, progress=None):
    """命令行和窗口共用入口；回调只发送数据，不操作窗口控件。"""
    progress = progress or (lambda event: None)
    source=Path(config['channel_dir'])
    if not source.is_absolute():
        source=FOLDER/source
    if not source.is_dir():
        raise ValueError(f'信道目录不存在：{source}')
    files=sorted(source.rglob(config['file_pattern']) if config['recursive'] else source.glob(config['file_pattern']))
    files=[p for p in files if p.is_file()]
    if not files:
        raise ValueError('未找到信道文件，请检查config.json中的目录与文件匹配规则。')
    progress({'kind':'start','total':len(files)})
    low,high=float(config['snr_min_db']),float(config['snr_max_db'])
    if not (np.isfinite(low) and np.isfinite(high) and low<high):
        raise ValueError('SNR范围无效。')
    seed=config['random_seed']
    if seed is None:
        seed=secrets.randbits(32)
    rng=np.random.default_rng(seed)
    snr_values=np.round(rng.uniform(low,high,len(files)),int(config['snr_decimal_places']))
    snr_values=np.clip(snr_values,low,high)
    active=active_model()
    checkpoint=project_path(config.get('checkpoint') or active['checkpoint'])
    torch.set_num_threads(4)
    predictor=Predictor(checkpoint,config['device'])
    target=predictor.checkpoint['config']['training'].get('target_metric','MeanDecodedBER')
    if target!='MeanCodedBER':
        raise ValueError('当前模型不是MeanCodedBER模型，停止批量预测。')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_root=Path(config.get('output_dir') or PROJECT/'outputs'/'batch_predictions')
    if not output_root.is_absolute():
        output_root=FOLDER/output_root
    output=output_root/stamp
    output.mkdir(parents=True,exist_ok=False)
    metadata={'random_seed':int(seed),'snr_source':'随机指定的验证条件，不是测量SNR',
              'model_path':str(checkpoint),'model_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'target_metric':target,'config':config,'files_count':len(files),
              'scope':'仅预测参数；未加噪声、未均衡仿真、未计算BER。'}
    write_json(output/'run_config.json',metadata)
    records=[]
    headers=['channel_name','channel_path','snr_db','mu','N1','N2','candidate_id','score','success','error']
    with (output/'batch_predictions.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=headers);writer.writeheader()
        for i,(path,snr) in enumerate(zip(files,snr_values)):
            row={'channel_name':path.name,'channel_path':str(path.resolve()),'snr_db':float(snr),
                 'mu':np.nan,'N1':np.nan,'N2':np.nan,'candidate_id':np.nan,'score':np.nan,'success':False,'error':''}
            try:
                h=read_field(path,config['channel_field'],predictor.checkpoint['config']['data']['hdf5_matlab_order'])
                p=predictor.predict(h,float(snr),top_k=1,already_cropped=config['already_cropped'],
                    main_path_matlab=None if config['already_cropped'] else config['main_path_matlab'],
                    allow_short=config['allow_short'])['selected']
                row.update(p,success=True)
                print(f"{i+1}/{len(files)} SNR={snr:.3f}dB -> mu={p['mu']:.7g}, N1={p['N1']}, N2={p['N2']}",flush=True)
            except Exception as error:
                row['error']=str(error)
                print(f'{i+1}/{len(files)} 失败：{path.name}：{error}',flush=True)
            records.append(row);writer.writerow(row);file.flush()
            progress({'kind':'row','index':i+1,'total':len(files),'row':row})
    mat_path=output/'batch_predictions.mat'
    save_mat(mat_path,records,metadata)
    # 回读MAT，核对SNR与参数没有因行顺序或维度变化而错配。
    restored=loadmat(mat_path)
    np.testing.assert_array_equal(restored['snr_db'].ravel(),snr_values)
    expected=np.array([[r[k] for k in ('mu','N1','N2')] for r in records])
    np.testing.assert_allclose(restored['parameters'],expected,equal_nan=True)
    assert restored['results'].shape==(len(files),1)
    success=sum(r['success'] for r in records)
    summary={'total':len(files),'successful':success,'failed':len(files)-success,
             'snr_min_actual_db':float(snr_values.min()),'snr_max_actual_db':float(snr_values.max()),
             'random_seed':int(seed),'mat_readback_verified':True,'mat_path':str(mat_path)}
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def main():
    config=json.loads((FOLDER/'config.json').read_text(encoding='utf-8-sig'))
    summary=run_batch(config)
    success=summary['successful']
    if success!=summary['total']:
        raise SystemExit('部分文件预测失败；MAT保留全部行，使用前请检查results(i).success与error。')


if __name__=='__main__':
    main()

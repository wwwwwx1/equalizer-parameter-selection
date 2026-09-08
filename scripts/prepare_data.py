"""在新电脑上建立训练索引和MeanCodedBER标签，不复制或修改原始信道。"""
import csv
import re
from pathlib import Path
import numpy as np

from nlms_dfe.common import ROOT, read_config, project_path, write_json
from nlms_dfe.progress import Progress
from nlms_dfe.data.io import read_field, load_candidates
from nlms_dfe.data.transforms import crop_channel


def matlab_text(value):
    value = np.asarray(value)
    if value.dtype.kind in 'ui':
        return ''.join(chr(int(x)) for x in value.ravel() if x)
    if value.dtype.kind in 'US':
        return ''.join(str(x) for x in value.ravel()).strip()
    raise ValueError('currName不是可读取的MATLAB字符串，请修改matlab_text适配器。')


def prepare(source, training):
    for key in ('channel_dir','result_dir'):
        if not source.get(key):
            raise ValueError(f'请先在configs/dataset.json填写{key}。')
    channel_root = project_path(source['channel_dir'])
    result_root = project_path(source['result_dir'])
    if not channel_root.is_dir() or not result_root.is_dir():
        raise ValueError('信道目录或结果目录不存在，请检查新电脑路径。')
    files = sorted(result_root.rglob(source['result_pattern']))
    if not files:
        raise ValueError('没有找到结果MAT；请检查result_pattern，例如grid_*.mat或*.mat。')
    out = project_path(source['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'准备目录非空：{out}。请另设新的output_dir，避免覆盖旧索引。')
    out.mkdir(parents=True, exist_ok=True)
    fields = source['fields']
    rows, provenance, reference = [], [], None
    csv_checks = 0
    progress=Progress("准备数据",len(files))
    progress.update(0,"开始读取首个信道与结果",force=True)
    for i, path in enumerate(files):
        try:
            read = lambda key: read_field(path, fields[key], source['hdf5_matlab_order'])
            name = matlab_text(read('channel_name'))
            # 结果有旧绝对路径时按文件名在channel_dir中配对，绝不去旧盘读取。
            name = name.replace('\\','/').split('/')[-1]
            channel_path = channel_root/name
            if not channel_path.is_file():
                matches = list(channel_root.rglob(name))
                if len(matches) != 1:
                    raise ValueError(f'无法唯一找到原始信道{name}，匹配数量{len(matches)}。')
                channel_path = matches[0]
            snr_values = np.asarray(read('snr'))
            if snr_values.size != 1:
                raise ValueError('每信道需要一个SNR标量。')
            snr = float(snr_values.item())
            parameters = np.stack([read(k).reshape(-1) for k in ('mu','N1','N2')], axis=1)
            count = len(parameters)
            if count != source['expected_candidates']:
                raise ValueError(f'候选数量为{count}，与配置不一致。')
            if reference is None:
                reference = parameters
                with (out/'candidates.csv').open('w',encoding='utf-8',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow(['candidate_id','mu','N1','N2'])
                    for k, p in enumerate(parameters):
                        writer.writerow([k+1,*p])
                load_candidates(out/'candidates.csv', count)
            np.testing.assert_allclose(parameters, reference, rtol=0, atol=0)
            mc = np.asarray(read('coded_mc'))
            # read_field已恢复MATLAB逻辑轴，应为[候选,重复次数]。
            if mc.ndim == 1 and len(mc) == count:
                mc = mc[:,None]
            if mc.ndim != 2 or mc.shape[0] != count or mc.shape[1] < 1:
                raise ValueError(f'coded_mc应为[候选,MC]，读取后是{mc.shape}；请核对轴顺序。')
            if not np.isfinite(snr) or not np.isfinite(mc).all() or np.any((mc<0)|(mc>1)):
                raise ValueError('SNR或BER存在缺失/非有限值，或BER不在[0,1]。')
            ber = mc.mean(axis=1)
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                with csv_path.open(encoding='utf-8-sig',newline='') as file:
                    table=sorted(csv.DictReader(file),key=lambda r:int(r['ParamIndex']))
                np.testing.assert_array_equal([int(r['ParamIndex']) for r in table],np.arange(1,count+1))
                np.testing.assert_allclose([float(r['MeanCodedBER']) for r in table],ber,atol=1e-12)
                csv_checks += 1
            h = read_field(channel_path,fields['channel'],source['hdf5_matlab_order'])
            cropped = crop_channel(h,training['data'])
            if not np.any(cropped):
                raise ValueError('裁剪后信道全零。')
            sample = f'channel_{i:07d}'
            label_path=out/f'{sample}.npz'
            np.savez(label_path,mean_coded_ber=ber,candidate_ids=np.arange(1,count+1))
            match=re.search(r'sig.start.(\d+)',name)
            group=f'sig_start_{match[1]}' if match else channel_path.stem
            rows.append(dict(sample_id=sample,channel_source_id=group,channel_path=str(channel_path.resolve()),
                             result_path=label_path.name,snr_db=snr,split='train'))
            provenance.append(dict(sample_id=sample,original_result=str(path.resolve()),
                                   original_channel=str(channel_path.resolve()),raw_shape=list(h.shape),
                                   cropped_shape=list(cropped.shape),mc_repeats=mc.shape[1]))
            progress.update(i+1,f'{name} -> {cropped.shape}，SNR={snr:g}')
        except Exception as error:
            write_json(out/'FAILED.json',{'file':str(path),'error':str(error),'completed':len(rows)})
            raise ValueError(f'数据准备停止：{path.name}：{error}。未跳过问题样本。') from error
    with (out/'index.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'provenance.json',provenance)
    report={'samples':len(rows),'num_candidates':len(reference),'target':'MeanCodedBER',
            'csv_crosschecks':csv_checks,'mode':'all_data','excluded_samples':0,
            'source_config':source,'crop_config':training['data']['crop'],
            'source_group_note':'名称分组仅作追溯；本次全量训练，不据此声称独立泛化。'}
    write_json(out/'audit.json',report)
    return report


def main():
    source=read_config('configs/dataset.json')
    training=read_config('configs/train.json')
    # 同步原始字段和读取轴配置，防止准备和训练的解释不一致。
    training['data']['fields']['channel']=source['fields']['channel']
    training['data']['hdf5_matlab_order']=source['hdf5_matlab_order']
    training['data']['num_candidates']=source['expected_candidates']
    training['data']['manifest']=str(Path(source['output_dir'])/'index.csv')
    training['data']['candidates']=str(Path(source['output_dir'])/'candidates.csv')
    report=prepare(source,training)
    write_json(ROOT/'configs/train.json',training)
    print('数据准备完成：',report['samples'],'个训练样本。')


if __name__=='__main__':
    main()

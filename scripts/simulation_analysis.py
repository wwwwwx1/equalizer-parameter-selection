"""可配置的预测/完整网格对比；区分网格内部共同噪声与独立仿真。"""
import csv
import html
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
from nlms_dfe.common import project_path,write_json
from nlms_dfe.progress import Progress


def read_hdf(file,node):
    if isinstance(node,h5py.Group):return {k:read_hdf(file,v) for k,v in node.items()}
    a=node[()]
    if h5py.check_dtype(ref=node.dtype):
        values=[read_hdf(file,file[r]) for r in a.ravel()]
        return values[0] if len(values)==1 else values
    if node.attrs.get('MATLAB_class')==b'char':return ''.join(chr(int(v)) for v in a.ravel())
    if a.dtype.names and {'real','imag'}<=set(a.dtype.names):a=a['real']+1j*a['imag']
    return a.T.squeeze()


def load_structs(path,key):
    if h5py.is_hdf5(path):
        with h5py.File(path) as file:
            if key not in file:raise ValueError(f'{path.name}缺少{key}，请保留仿真明细。')
            group=file[key]
            ignored={'resultTable','resultTableSorted','bestResult'}
            if h5py.check_dtype(ref=group['channel_name'].dtype):
                n=group['channel_name'].size
                rows=[{k:read_hdf(file,file[v[()].ravel()[i]]) for k,v in group.items() if k not in ignored} for i in range(n)]
            else:
                # MATLAB保存单个struct时字段可能直接是数值/char，而非对象引用。
                rows=[{k:read_hdf(file,v) for k,v in group.items() if k not in ignored}]
            metadata={k:read_hdf(file,file[k]) for k in ('M','MC','Rb','Rs','beta','span','fc','fs','fss','num','num_info','train_ratio','msg_source','code_data','ss') if k in file}
            return rows,metadata
    data=loadmat(path,simplify_cells=True)
    rows=data.get(key)
    if rows is None:raise ValueError(f'缺少{key}。')
    if isinstance(rows,dict):rows=[rows]
    return list(rows),{k:data[k] for k in ('M','MC','Rb','Rs','beta','span','fc','fs','fss','num','num_info','train_ratio','msg_source','code_data','ss') if k in data}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(job):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=project_path(job['output'])
    if output.exists() and any(output.iterdir()):raise ValueError('分析目录非空，请另选新目录。')
    output.mkdir(parents=True,exist_ok=True)
    pred_path=project_path(job['prediction_mat']);sim_path=project_path(job['simulation_mat']);grid_path=project_path(job['grid_mat'])
    if h5py.is_hdf5(pred_path):
        with h5py.File(pred_path) as file:
            p={k:read_hdf(file,file[k]) for k in ('channel_names','channel_paths','snr_db','mu','N1','N2','success') if k in file}
    else:
        p=loadmat(pred_path,simplify_cells=True)
    names=np.atleast_1d(p['channel_names']).astype(str).tolist()
    snr=np.atleast_1d(p['snr_db']).astype(float)
    predicted=np.column_stack([np.atleast_1d(p[k]).astype(float) for k in ('mu','N1','N2')])
    if len(set(names))!=len(names):raise ValueError('输入清单信道名重复，需先明确唯一对应关系。')
    sim,sm=load_structs(sim_path,'simulation_results');grid,gm=load_structs(grid_path,'all_grid_results')
    sim_map={str(r['channel_name']):r for r in sim};grid_map={str(r['channel_name']):r for r in grid}
    if len(sim_map)!=len(sim) or len(grid_map)!=len(grid):raise ValueError('仿真结果信道名重复。')
    meta_checks={k:bool(np.array_equal(sm[k],gm[k])) for k in sm.keys()&gm.keys()}
    disagreements=[k for k,v in meta_checks.items() if not v and k!='MC']
    if disagreements:raise ValueError('两套仿真协议/发送序列不一致：'+','.join(disagreements))
    missing_meta=sorted(set(['M','Rb','Rs','fc','fs','fss','num','num_info','msg_source','code_data'])-(sm.keys()&gm.keys()))
    fixed=job.get('fixed_parameters')
    fixed_origin='用户指定固定参数'
    if fixed is None:
        import torch
        checkpoint=torch.load(project_path(job['checkpoint']),map_location='cpu',weights_only=True)
        fixed=checkpoint['candidates'][checkpoint['fixed_index']].tolist();fixed_origin='模型训练数据选定的固定参数'
    progress=Progress('读取并对齐仿真结果',len(names));progress.update(0)
    per=[];excluded=[];grid_means=[];grid_decoded=[];candidates=None
    independent=[];common=[];optimal=[];baseline=[];decoded_present=True;noise_checks=[]
    success_flags=np.atleast_1d(p.get('success',np.ones(len(names),dtype=bool)))
    for i,name in enumerate(names):
        s,g=sim_map.get(name),grid_map.get(name)
        reason=''
        if s is None or g is None:reason='缺少对应仿真记录'
        elif not success_flags[i] or not bool(s['simulation_success']) or not bool(g['simulation_success']):reason='预测或仿真失败'
        if reason:
            excluded.append({'channel_name':name,'reason':reason});progress.update(i+1,reason);continue
        if not np.isclose(float(s['snr_db']),snr[i],rtol=0,atol=1e-8) or not np.isclose(float(g['snr_db']),snr[i],rtol=0,atol=1e-8):raise ValueError(name+'的SNR不匹配。')
        if 'H' in s and 'H' in g and not np.array_equal(s['H'],g['H']):raise ValueError(name+'的保存信道H不一致。')
        pars=np.column_stack([np.atleast_1d(g[k]) for k in ('param_delt','param_N1','param_N2')]).astype(float)
        if candidates is None:candidates=pars
        if not np.array_equal(pars,candidates):raise ValueError('不同信道的候选表不一致。')
        if len(np.unique(pars,axis=0))!=len(pars):raise ValueError('候选参数重复。')
        selected=np.flatnonzero(np.all(np.isclose(pars,predicted[i],rtol=0,atol=1e-7),axis=1))
        fixed_id=np.flatnonzero(np.all(np.isclose(pars,fixed,rtol=0,atol=1e-7),axis=1))
        if len(selected)!=1 or len(fixed_id)!=1:raise ValueError('推荐参数或固定参数无法唯一匹配网格。')
        if not np.allclose([s[k] for k in ('mu','N1','N2')],predicted[i],rtol=0,atol=1e-7):raise ValueError('预测仿真实际参数与输入清单不一致。')
        if 'coded_ber_after_mc' not in s or 'coded_ber_after_mc' not in g:
            raise ValueError('分析需要保留“译码前MC BER”明细，请重新选择保存内容。')
        C=np.asarray(g['coded_ber_after_mc'],dtype=float).reshape(len(pars),-1)
        S=np.asarray(s['coded_ber_after_mc'],dtype=float).reshape(-1)
        if not np.isfinite(C).all() or not np.isfinite(S).all() or np.any((C<0)|(C>1)) or np.any((S<0)|(S>1)):raise ValueError('BER非法。')
        mean=C.mean(1);ix=int(selected[0]);fi=int(fixed_id[0]);bi=int(mean.argmin())
        grid_means.append(mean);independent.append(float(S.mean()));common.append(float(mean[ix]));optimal.append(float(mean[bi]));baseline.append(float(mean[fi]))
        same_before=None
        if 'coded_ber_before_mc' in s and 'coded_ber_before_mc' in g:
            same_before=bool(np.array_equal(s['coded_ber_before_mc'],g['coded_ber_before_mc']))
        noise_checks.append(same_before)
        row={'channel_name':name,'snr_db':snr[i],'predicted_id':ix+1,'mu':pars[ix,0],'N1':int(pars[ix,1]),'N2':int(pars[ix,2]),
             'grid_best_id':bi+1,'grid_best_mu':pars[bi,0],'grid_best_N1':int(pars[bi,1]),'grid_best_N2':int(pars[bi,2]),
             'prediction_independent_coded':S.mean(),'prediction_common_noise_coded':mean[ix],'grid_best_coded':mean[bi],
             'fixed_coded':mean[fi],'regret':mean[ix]-mean[bi],'rank_tie_aware':1+int(np.sum(mean<mean[ix]-1e-12)),
             'coded_before_mc_equal':same_before}
        if 'decoded_ber_after_mc' in s and 'decoded_ber_after_mc' in g:
            D=np.asarray(g['decoded_ber_after_mc'],dtype=float).reshape(len(pars),-1)
            SD=np.asarray(s['decoded_ber_after_mc'],dtype=float)
            if not np.isfinite(D).all() or not np.isfinite(SD).all() or np.any((D<0)|(D>1)) or np.any((SD<0)|(SD>1)):
                raise ValueError('辅助Decoded BER含非法值。')
            row.update(prediction_independent_decoded=np.asarray(s['decoded_ber_after_mc']).mean(),
                       prediction_common_noise_decoded=D[ix].mean(),coded_optimum_decoded=D[bi].mean(),fixed_decoded=D[fi].mean())
            grid_decoded.append(D.mean(1))
        else:decoded_present=False
        per.append(row);progress.update(i+1,name)
    if not per:raise ValueError('没有成功且可对应的仿真记录。')
    progress.update(len(names),'对齐完成，开始出图',force=True)
    # 混合保存方案下让CSV字段一致，缺失Decoded不填造数值。
    all_fields=list(dict.fromkeys(k for r in per for k in r))
    per=[{k:r.get(k,'') for k in all_fields} for r in per]
    write_csv(output/'逐信道对比.csv',per);write_csv(output/'排除记录.csv',excluded)
    write_csv(output/'异常信道排序.csv',sorted(per,key=lambda r:r['regret'],reverse=True))
    strategies=[]
    for name,values in [('预测参数（共同噪声）',common),('预测参数（实际仿真）',independent),('网格最小参考',optimal),(fixed_origin,baseline)]:
        values=np.asarray(values)
        strategies.append({'strategy':name,'mean_coded_ber':float(values.mean()),'median_coded_ber':float(np.median(values)),
                           'p90_coded_ber':float(np.quantile(values,.9)),'channels':len(values)})
    write_csv(output/'策略汇总.csv',strategies)
    all_grid=np.stack(grid_means)
    ranking=all_grid.mean(0).argsort()
    write_csv(output/'网格固定参数排名.csv',[{'rank':rank+1,'candidate_id':int(k+1),'mu':candidates[k,0],
               'N1':int(candidates[k,1]),'N2':int(candidates[k,2]),'batch_mean_coded_ber':float(all_grid[:,k].mean())}
              for rank,k in enumerate(ranking)])
    np.savez_compressed(output/'analysis_arrays.npz',grid_mean_coded=all_grid,parameters=candidates,
                        prediction_common=common,prediction_independent=independent,fixed=baseline,grid_best=optimal)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    figures=output/'figures';figures.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(figures/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    x=np.arange(1,len(per)+1)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    for label,values,style in [('预测：共同噪声',common,'-o'),('预测：实际仿真',independent,'--s'),('固定参数',baseline,'-^'),('网格最小参考',optimal,':d')]:
        ax.plot(x,np.array(values)*100,style,label=label,markersize=3)
    ax.set(xlabel='信道序号（输入顺序）',ylabel='MeanCodedBER（%）',title='逐信道对比：共同噪声与实际预测仿真分开显示');ax.legend();save(fig,'01_coded_comparison')
    delta=np.array(common)-baseline
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    ax.bar(x,delta*100,color=np.where(delta<=0,'#397DAE','#BB675B'));ax.axhline(0,color='gray')
    ax.set(xlabel='信道序号',ylabel='预测 − 固定（百分点）',title='网格内部共同噪声比较：负值表示推荐参数更好');save(fig,'02_paired_difference')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.scatter([r['snr_db'] for r in per],[r['regret']*100 for r in per],color='#397DAE')
    ax.set(xlabel='SNR（dB）',ylabel='相对网格最小值的BER差距（百分点）',title='不同SNR下的选参误差');save(fig,'03_snr_regret')
    if decoded_present:
        fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
        for label,key in [('预测：共同噪声','prediction_common_noise_decoded'),('预测：实际仿真','prediction_independent_decoded'),('固定参数','fixed_decoded')]:
            ax.plot(x,[float(r[key])*100 for r in per],'-o',label=label,markersize=3)
        ax.set(xlabel='信道序号',ylabel='MeanDecodedBER（%）',title='辅助指标：译码后误码率（不改变MeanCodedBER选参目标）');ax.legend();save(fig,'04_decoded_comparison')
    fig,(ax,zero)=plt.subplots(2,1,figsize=(11,6.2),sharex=True,gridspec_kw={'height_ratios':[5,1]},layout='constrained')
    for j,(label,values) in enumerate([('预测：共同噪声',common),('固定参数',baseline),('网格最小参考',optimal)]):
        a=np.array(values);positive=np.where(a>0,a,np.nan)
        line,=ax.plot(x,positive,'-o',label=label,markersize=3,color=['#1f77b4','#2ca02c','#d62728'][j])
        zero.scatter(x[a==0],np.full(np.sum(a==0),j),color=line.get_color(),s=24)
    ax.set_yscale('log');ax.set(ylabel='MeanCodedBER（对数）',title='低误码细节：零值单独标注，不替换为非零数值');ax.legend()
    if not np.any(np.array(common+baseline+optimal)>0):
        ax.set_ylim(1e-6,1);ax.text(.5,.5,'所有观测BER均为0，见下方零值标记',transform=ax.transAxes,ha='center')
    zero.set(xlabel='信道序号',yticks=[0,1,2],yticklabels=['预测零值','固定零值','网格零值'],ylim=(-.6,2.6))
    save(fig,'05_low_ber_detail')
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    for ax,key,best_key in zip(axes,['mu','N1','N2'],['grid_best_mu','grid_best_N1','grid_best_N2']):
        ax.plot(x,[r[key] for r in per],'-o',label='预测参数',markersize=3)
        ax.plot(x,[r[best_key] for r in per],'--s',label='一个网格最优组合',markersize=3,color='#d62728')
        ax.set_ylabel(key)
    axes[0].set_title('参数组合对照：并列最优时显示其中一组')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper right',ncol=2)
    axes[-1].set_xlabel('信道序号')
    save(fig,'06_parameter_comparison')
    audit={'input_channels':len(names),'analyzed_channels':len(per),'excluded_channels':len(excluded),
           'protocol_checks':meta_checks,'missing_protocol_fields':missing_meta,'fixed_parameters':list(map(float,fixed)),
           'fixed_origin':fixed_origin,'before_mc_equal_count':sum(v is True for v in noise_checks),
           'note':'均衡前BER相等也不能单独证明噪声逐位相同。主要配对比较从同一网格原始MC矩阵抽取预测候选。',
           'prediction_mat':str(pred_path),'simulation_mat':str(sim_path),'grid_mat':str(grid_path)}
    write_json(output/'数据核验.json',audit)
    table='\n'.join(f"| {r['strategy']} | {r['mean_coded_ber']:.6f} |" for r in strategies)
    report=f'''# 仿真对比分析

输入{len(names)}个信道，实际分析{len(per)}个，排除{len(excluded)}个。排除原因另存CSV，没有静默删除失败样本。

| 策略 | 平均MeanCodedBER |
|---|---:|
{table}

固定参数为{list(map(float,fixed))}，来源：{fixed_origin}。网格固定参数排名属于本批次事后统计，不冒充训练集选定基线。

共同噪声比较从同一网格矩阵抽取推荐候选，与固定参数和网格最小值比较。实际预测仿真单独保留，不因为两脚本种子相同就假设每次噪声相同。

主要比较中：优于固定{int(np.sum(delta<-1e-12))}个、相同{int(np.sum(abs(delta)<=1e-12))}个、差于固定{int(np.sum(delta>1e-12))}个。

未保存而无法核对的协议字段：{missing_meta}。本次是给定文件的描述性分析，不自动证明未见信道泛化或统计显著性；小样本、同源相关性和固定报文仍须考虑。

图与数据均覆盖实际分析的全部信道，BER使用线性轴，零值保留。图为PNG/PDF/SVG。MeanDecodedBER仅在两份文件均保留原始数组时输出为辅助图。
'''
    (output/'分析报告.md').write_text(report,encoding='utf-8')
    images=''.join(f'<img src="figures/{f.name}" style="max-width:100%;margin:12px 0">' for f in sorted(figures.glob('*.png')))
    (output/'分析报告.html').write_text('<meta charset="utf-8"><body style="max-width:1100px;margin:30px auto;font-family:sans-serif"><pre style="white-space:pre-wrap">'+html.escape(report)+'</pre>'+images+'</body>',encoding='utf-8')
    return {'action':'simulation_analysis','output':str(output),'message':f'分析完成：{len(per)}个信道，已输出图表、CSV和HTML报告。'}

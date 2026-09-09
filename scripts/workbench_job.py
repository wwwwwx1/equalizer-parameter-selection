"""统一窗口的独立后台进程：配置快照驱动任务，日志以UTF-8传回。"""
import sys
from pathlib import Path
from nlms_dfe.common import ROOT,read_config,write_json,project_path
from nlms_dfe.progress import event


def run(job):
    action=job['action']
    if action=='ablation':
        from scripts.ablation_workbench import run as run_ablation
        return run_ablation(job)
    if action=='matlab_simulation':
        from scripts.matlab_bridge import run_matlab
        return run_matlab(job)
    if action=='simulation_analysis':
        from scripts.simulation_analysis import analyze
        return analyze(job)
    if action=='prepare':
        from scripts.prepare_data import prepare
        source,config=job['source'],job['config']
        config['data']['fields']['channel']=source['fields']['channel']
        config['data']['hdf5_matlab_order']=source['hdf5_matlab_order']
        config['data']['num_candidates']=source['expected_candidates']
        config['data']['manifest']=str(project_path(source['output_dir'])/'index.csv')
        config['data']['candidates']=str(project_path(source['output_dir'])/'candidates.csv')
        report=prepare(source,config)
        path=project_path(source['output_dir'])/'train_config.json'
        write_json(path,config)
        return {'action':action,'samples':report['samples'],'config_path':str(path),
                'output':str(path.parent),'message':f"数据准备完成，共{report['samples']}个样本。训练页已连接新索引。"}
    if action=='train':
        from nlms_dfe.training.engine import train
        config=job['config']
        if job.get('split_ratio'):
            from nlms_dfe.data.splitting import write_split
            output=project_path(config['training']['output'])
            manifest=write_split(project_path(config['data']['manifest']),output.with_name(output.name+'_split'),config['training']['seed'])
            config['data']['manifest']=str(manifest)
            config['training']['mode']='holdout'
        checkpoint=train(config)
        metrics=None
        if config['training'].get('mode')=='holdout':
            from nlms_dfe.training.engine import evaluate
            metrics=evaluate(checkpoint,checkpoint.parent/'evaluation')
        try:relative=str(checkpoint.relative_to(ROOT))
        except ValueError:relative=str(checkpoint)
        if job.get('publish_model',True):
            write_json(ROOT/'configs/inference.json',{'checkpoint':relative,'target_metric':config['training']['target_metric'],
                       'label':f"MeanCodedBER · {config['model']['layers']}层时间网络"})
        warning=''
        try:
            from scripts.plot_training import plot
            plot(checkpoint.parent)
        except Exception as error:
            warning=f'模型已保存，但出图失败：{error}'
            print(warning,flush=True)
        return {'action':action,'checkpoint':str(checkpoint),'output':str(checkpoint.parent),
                'message':warning or (f"训练完成：独立测试BER={metrics['selected_ber_macro']:.6g}，固定参数BER={metrics['fixed_ber_macro']:.6g}；模型已用于预测。" if metrics else '训练和出图完成；单信道、批量预测已切换为新模型。')}
    if action=='single':
        import torch
        from scipy.io import savemat
        from nlms_dfe.data.io import read_field
        from nlms_dfe.predict import Predictor
        torch.set_num_threads(4)
        model=Predictor(project_path(job['checkpoint']))
        if model.checkpoint['config']['training'].get('target_metric','MeanDecodedBER')!='MeanCodedBER':
            raise ValueError('请选择MeanCodedBER模型，避免使用旧版Decoded目标。')
        h=read_field(job['channel'],job['field'],model.checkpoint['config']['data']['hdf5_matlab_order'])
        result=model.predict(h,job['snr'],already_cropped=job['cropped'],
                             main_path_matlab=None if job['cropped'] else job['center'],allow_short=job['allow_short'])
        folder=project_path(job['output']);folder.mkdir(parents=True,exist_ok=False)
        result.update(channel_path=job['channel'],snr_db=job['snr'],checkpoint=job['checkpoint'])
        write_json(folder/'prediction.json',result)
        savemat(folder/'prediction.mat',{key:result['selected'][key] for key in ('mu','N1','N2','candidate_id')})
        p=result['selected']
        return {'action':action,'output':str(folder),'prediction':p,
                'message':f"μ={p['mu']:.7g}，N1={p['N1']}，N2={p['N2']}；MAT已保存。"}
    if action=='batch':
        from scripts.batch_service import run_batch
        def callback(item):
            if item['kind']=='start':
                event('progress',stage='批量预测',current=0,total=item['total'],percent=0,detail='加载模型')
            else:
                row=item['row']
                event('progress',stage='批量预测',current=item['index'],total=item['total'],
                      percent=100*item['index']/item['total'],detail=row['channel_name'])
                event('prediction',row=row)
        # 显式传入窗口选定的模型，不更改全局默认配置。
        summary=run_batch(job['batch'],callback)
        return {'action':action,'output':str(Path(summary['mat_path']).parent),'summary':summary,
                'message':f"批量完成：成功{summary['successful']}，失败{summary['failed']}；MAT已保存。"}
    raise ValueError('未知任务：'+action)


if __name__=='__main__':
    result=run(read_config(sys.argv[1]))
    event('result',**result)

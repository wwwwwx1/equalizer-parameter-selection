"""独立临时样本检查四种后台任务及进度事件，不改变现有模型。"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat,loadmat
from nlms_dfe.common import ROOT,read_config,write_json


class WorkbenchTests(unittest.TestCase):
    def test_all_four_modes_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);channels=folder/'channels';results=folder/'results'
            channels.mkdir();results.mkdir()
            config_path='configs/train_window.json' if (ROOT/'configs/train_window.json').exists() else 'configs/train.json'
            config=read_config(config_path)
            config['data'].update(num_candidates=6,cache_features=False,temporal_pool=2)
            config['data']['crop'].update(already_cropped=False,time_start_matlab=2,time_stop_matlab=10,
                                          main_path_matlab=5,delay_before=4,delay_after=4)
            config['model'].update(d_model=16,heads=2,layers=1,ffn_dim=32,delay_bins=2,candidate_dim=8)
            config['training'].update(epochs=1,batch_size=2,workers=0,amp=False,device='cpu',mode='all_data',
                                       output=str(folder/'training'),target_metric='MeanCodedBER')
            rng=np.random.default_rng(2)
            for i in range(3):
                savemat(channels/f'channel{i}.mat',{'h_vary':rng.normal(size=(12,9))+1j*rng.normal(size=(12,9))})
                savemat(results/f'grid_{i}.mat',{'currName':f'channel{i}.mat','currSNR':float(i),
                        'param_delt':np.linspace(.1,.6,6),'param_N1':np.arange(1,7),'param_N2':np.arange(6),
                        'coded_ber_after_mc':np.linspace(.01,.3,6)[:,None]*np.ones((1,2))})
            source={'channel_dir':str(channels),'result_dir':str(results),'result_pattern':'grid_*.mat',
                    'output_dir':str(folder/'prepared'),'expected_candidates':6,'hdf5_matlab_order':True,
                    'fields':{'channel':'h_vary','channel_name':'currName','snr':'currSNR','mu':'param_delt',
                              'N1':'param_N1','N2':'param_N2','coded_mc':'coded_ber_after_mc'}}
            def execute(job):
                path=folder/(job['action']+'.json');write_json(path,job)
                env=os.environ.copy();env.update(PYTHONUTF8='1',NLMS_UI_EVENTS='1',NLMS_PROGRESS_FILE=str(folder/'progress.json'))
                proc=subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.workbench_job',str(path)],cwd=ROOT,
                                    env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                events=[json.loads(line[len('@@NLMS@@'):]) for line in proc.stdout.splitlines() if line.startswith('@@NLMS@@')]
                self.assertEqual(events[-1]['kind'],'result')
                return events,events[-1]
            events,result=execute({'action':'prepare','source':source,'config':config})
            self.assertEqual(result['samples'],3)
            self.assertTrue(any(e['kind']=='progress' and e['current']==3 for e in events))
            config=read_config(result['config_path'])
            events,result=execute({'action':'train','config':config,'publish_model':False})
            scans=[e for e in events if e['kind']=='progress' and e['stage']=='扫描训练数据集']
            self.assertEqual(scans[0]['current'],0);self.assertEqual(scans[-1]['current'],3)
            self.assertEqual(scans[-1]['percent'],100)
            checkpoint=result['checkpoint']
            self.assertTrue((folder/'training/figures/training_curves.png').exists())
            _,single=execute({'action':'single','checkpoint':checkpoint,'channel':str(channels/'channel0.mat'),
                               'snr':1.2,'field':'h_vary','center':5,'cropped':False,'allow_short':False,'output':str(folder/'single')})
            self.assertTrue((folder/'single/prediction.mat').exists())
            events,batch=execute({'action':'batch','batch':{'channel_dir':str(channels),'file_pattern':'*.mat','recursive':False,
                                   'channel_field':'h_vary','checkpoint':checkpoint,'main_path_matlab':5,'already_cropped':False,
                                   'allow_short':False,'snr_min_db':-5,'snr_max_db':5,'snr_decimal_places':3,'random_seed':42,
                                   'device':'cpu','output_dir':str(folder/'batch')}})
            self.assertEqual(batch['summary']['successful'],3)
            data=loadmat(batch['summary']['mat_path'],simplify_cells=True)
            self.assertEqual(len(data['results']),3)
            self.assertTrue(np.all((data['snr_db']>=-5)&(data['snr_db']<=5)))
            self.assertEqual(sum(e['kind']=='prediction' for e in events),3)


if __name__=='__main__':unittest.main()

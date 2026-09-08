"""独立于旧电脑数据的临时合成样本检查；临时权重在测试结束后清理。"""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import h5py
import torch
from scipy.io import savemat,loadmat

from nlms_dfe.common import read_config
from nlms_dfe.data.transforms import crop_channel
from nlms_dfe.data.io import read_field,load_manifest
from nlms_dfe.predict import Predictor
from nlms_dfe.training.engine import train,evaluate
from scripts.prepare_data import prepare,matlab_text
from scripts.plot_training import plot


class PortableTests(unittest.TestCase):
    def test_default_main_path_501(self):
        config=read_config('configs/train.json')['data']
        h=np.zeros((4000,751),dtype=np.complex64)
        h[:,500]=1+2j
        cut=crop_channel(h,config)
        self.assertEqual(cut.shape,(3375,501))
        np.testing.assert_equal(cut[:,250],1+2j)
        with self.assertRaises(ValueError):
            crop_channel(h[:3250],config)
        config['crop']['allow_short']=True
        self.assertEqual(crop_channel(h[:3250],config).shape,(2625,501))

    def test_missing_paths_fail_clearly(self):
        source=read_config('configs/dataset.json')
        source.update(channel_dir='',result_dir='')
        with self.assertRaisesRegex(ValueError,'channel_dir'):
            prepare(source,read_config('configs/train.json'))

    def test_matlab_v73_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.mat'
            expected=np.arange(18).reshape(6,3)/100
            with h5py.File(path,'w') as file:
                file.create_dataset('coded_ber_after_mc',data=expected.T)
                file.create_dataset('currName',data=np.array([ord(c) for c in 'new.mat'],dtype=np.uint16)[:,None])
            np.testing.assert_equal(read_field(path,'coded_ber_after_mc'),expected)
            self.assertEqual(matlab_text(read_field(path,'currName')),'new.mat')

    def test_prepare_train_plot_predict(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            channels=root/'channels';channels.mkdir()
            results=root/'results';results.mkdir()
            rng=np.random.default_rng(1)
            for i in range(3):
                h=rng.normal(size=(20,9))+1j*rng.normal(size=(20,9))
                savemat(channels/f'ch{i}.mat',{'h_vary':h})
                ber=np.linspace(.01,.3,6)[:,None]*np.ones((1,3))
                savemat(results/f'grid_{i}.mat',{'currName':f'ch{i}.mat','currSNR':i,
                        'param_delt':np.linspace(.1,.6,6),'param_N1':np.arange(1,7),'param_N2':np.arange(6),
                        'coded_ber_after_mc':ber})
                with (results/f'grid_{i}.csv').open('w',newline='') as file:
                    writer=csv.writer(file);writer.writerow(['ParamIndex','MeanCodedBER'])
                    writer.writerows((k+1,float(ber[k].mean())) for k in reversed(range(6)))
            source=read_config('configs/dataset.json')
            source.update(channel_dir=str(channels),result_dir=str(results),output_dir=str(root/'prepared'),expected_candidates=6)
            config=read_config('configs/train.json')
            config['data']['crop'].update(time_start_matlab=2,time_stop_matlab=18,main_path_matlab=5,delay_before=4,delay_after=4)
            report=prepare(source,config)
            self.assertEqual(report['samples'],3)
            self.assertEqual(report['csv_crosschecks'],3)
            config['data'].update(manifest=str(root/'prepared/index.csv'),candidates=str(root/'prepared/candidates.csv'),num_candidates=6,temporal_pool=2)
            config['model'].update(d_model=16,heads=2,layers=1,ffn_dim=32,delay_bins=2,candidate_dim=8)
            config['training'].update(epochs=1,batch_size=2,amp=False,device='cpu',output=str(root/'run'))
            model_path=train(config)
            predictor=Predictor(model_path)
            self.assertEqual(len(predictor.checkpoint['manifest_snapshot']),3)
            self.assertEqual(predictor.checkpoint['config']['training']['target_metric'],'MeanCodedBER')
            prediction=predictor.predict(h,2,already_cropped=False,main_path_matlab=5)
            self.assertEqual(prediction['target_metric'],'MeanCodedBER')
            self.assertTrue(1<=prediction['selected']['candidate_id']<=6)
            with self.assertRaises(ValueError):
                evaluate(model_path,root/'not_a_test')
            plot(root/'run')
            for suffix in ('png','pdf','svg'):
                self.assertTrue((root/f'run/figures/training_curves.{suffix}').is_file())
            with self.assertRaisesRegex(ValueError,'非空'):
                prepare(source,config)


if __name__=='__main__':
    unittest.main()

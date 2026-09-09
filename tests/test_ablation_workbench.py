"""临时合成数据验证并发与组外评估，不作为通信性能结果。"""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from nlms_dfe.common import read_config
from scripts.ablation_workbench import run

class MultiModelTests(unittest.TestCase):
    def test_concurrent_training_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=read_config('configs/train_window.json')
            c['data']['crop'].update(already_cropped=True,delay_before=8,delay_after=8)
            c['data'].update(temporal_pool=2,num_candidates=3,cache_features=False)
            c['model'].update(d_model=16,heads=2,layers=0,ffn_dim=32,delay_bins=2,candidate_dim=8,dropout=0)
            c['training'].update(epochs=1,batch_size=2,workers=0,device='cpu',amp=False)
            c['evaluation']['bootstrap_repeats']=10
            candidates=root/'candidates.csv';candidates.write_text('candidate_id,mu,N1,N2\n1,.01,2,1\n2,.02,3,2\n3,.03,4,3\n')
            manifest=root/'input.csv'
            with manifest.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['sample_id','channel_source_id','channel_path','split','snr_db'])
                for i in range(6):
                    path=root/f'{i}.npz';rng=np.random.default_rng(i)
                    np.savez(path,h_vary=rng.normal(size=(10,17))+1j*rng.normal(size=(10,17)),mean_coded_ber=[.1,.2,.3],candidate_ids=[1,2,3])
                    w.writerow([str(i),str(i),str(path),'train',i])
            c['data'].update(manifest=str(manifest),candidates=str(candidates))
            out=root/'results'
            result=run(dict(config=c,output=str(out),variants=['full_current','no_delay_cnn_current'],seeds=[42],concurrency=2,split_seed=42,mode='holdout'))
            self.assertIn('成功2/2',result['message'])
            self.assertTrue((out/'comparison.csv').exists())
            for name in ['full_current','no_delay_cnn_current']:
                folder=out/f'{name}_seed42'
                self.assertTrue((folder/'figures/training_curves.png').exists())
                self.assertEqual(read_config(folder/'result.json')['scope'],'组外测试')

if __name__=='__main__':unittest.main()

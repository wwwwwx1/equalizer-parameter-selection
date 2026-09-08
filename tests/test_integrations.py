"""配置编辑、MATLAB运行副本和可变样本数结果分析的独立检查。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from nlms_dfe.common import ROOT,read_config
from scripts.matlab_bridge import prepare_script
from scripts.workbench_extensions import validate_model_config
from scripts.simulation_analysis import analyze


class IntegrationTests(unittest.TestCase):
    def config(self):
        return read_config('configs/train_window.json' if (ROOT/'configs/train_window.json').exists() else 'configs/train.json')

    def test_config_validation(self):
        c=self.config();c['model'].update(d_model=128,heads=8,layers=6)
        validate_model_config(c)
        c['model']['heads']=7
        with self.assertRaises(ValueError):validate_model_config(c)
        c['model']['heads']=0
        with self.assertRaises(ValueError):validate_model_config(c)

    def test_matlab_script_is_only_patched_copy(self):
        cfg=read_config('configs/matlab_workbench.json')
        original=ROOT/cfg['grid_script'];before=original.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.update(mode='grid',script=str(original),input_mat='a.mat',output_mat='b.mat',mc=1,center=501,max_channels=1,parallel=False,status_file='status.json')
            script=prepare_script(cfg,Path(tmp));text=script.read_text(encoding='utf-8')
            self.assertIn('predictionMatFile = WBconfig.input_mat',text)
            self.assertIn('Index0_center = WBconfig.center',text)
            self.assertIn('wb_save(allResultMatFile',text)
            self.assertIn('wb_finish(WBconfig,all_grid_results)',text)
            self.assertNotIn('        parfor idx',text)
            self.assertEqual(original.read_bytes(),before)

    def test_eight_page_ui_and_editor_apply(self):
        import tkinter as tk
        from scripts.workbench_gui import Workbench
        with tempfile.TemporaryDirectory() as tmp:
            root=tk.Tk();root.withdraw();app=Workbench(root)
            self.assertEqual(len(app.tabs.tabs()),8)
            ext=app.extensions
            ext.edit_config['model'].update(layers=6,d_model=128,heads=8)
            with patch('scripts.workbench_extensions.ROOT',Path(tmp)):
                ext.apply()
                self.assertEqual(app.layers.get(),'6')
                self.assertEqual(app.config['model']['d_model'],128)
                self.assertTrue(Path(app.config_path.get()).is_file())
            ext.sim['prediction']['input'].set('input.mat');ext.sim['prediction']['output'].set('output.mat')
            ext.sim['prediction']['save']['mc_decoded'].set(False)
            job=ext.matlab_job('prediction')
            self.assertFalse(job['matlab']['save_options']['mc_decoded'])
            self.assertEqual(job['matlab']['mode'],'prediction')
            root.destroy()

    def test_analysis_arbitrary_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            pars=np.array([[.2,10,20],[.3,20,30],[.4,30,40]])
            names=['a.mat','b.mat'];sim=[];grid=[]
            for i,name in enumerate(names):
                C=np.array([[.1,.2],[.05,.1],[.3,.4]])+i*.01
                common=dict(channel_name=name,snr_db=i,simulation_success=True)
                grid.append(dict(**common,param_delt=pars[:,0],param_N1=pars[:,1],param_N2=pars[:,2],
                                 coded_ber_after_mc=C,decoded_ber_after_mc=C/2))
                sim.append(dict(**common,mu=.3,N1=20,N2=30,coded_ber_after_mc=C[1]+.01,decoded_ber_after_mc=C[1]/2))
            savemat(folder/'pred.mat',dict(channel_names=np.array(names,dtype=object),snr_db=[0,1],mu=[.3,.3],N1=[20,20],N2=[30,30],success=[1,1]))
            savemat(folder/'sim.mat',dict(simulation_results=np.array(sim,dtype=object),MC=2))
            savemat(folder/'grid.mat',dict(all_grid_results=np.array(grid,dtype=object),MC=2))
            job=dict(prediction_mat=str(folder/'pred.mat'),simulation_mat=str(folder/'sim.mat'),grid_mat=str(folder/'grid.mat'),
                     fixed_parameters=[.2,10,20],output=str(folder/'report'))
            result=analyze(job)
            self.assertEqual(json.loads((folder/'report/数据核验.json').read_text(encoding='utf-8'))['analyzed_channels'],2)
            self.assertTrue((folder/'report/分析报告.html').exists())
            self.assertEqual(len(list((folder/'report/figures').glob('*.png'))),6)
            self.assertEqual(result['action'],'simulation_analysis')


if __name__=='__main__':unittest.main()

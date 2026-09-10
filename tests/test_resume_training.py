"""验证完整轮次续训与不中断训练的权重一致。"""
import copy
import os
import unittest
from unittest.mock import patch
import torch
from tests.test_pipeline import PipelineTests
from nlms_dfe.training.engine import train

class ResumeTests(unittest.TestCase):
    def test_epoch_resume(self):
        from nlms_dfe.common import read_config
        config=read_config('configs/train_window.json');config['data']['fields']['ber']='ber';config['training']['mode']='holdout'
        fixture=PipelineTests()
        with patch('tests.test_pipeline.read_config',return_value=config):fixture.setUp()
        try:
            fixture.run_round_trip('cpu')
            root=fixture.root
            base=torch.load(root/'run/last.pt',weights_only=True)
            c=copy.deepcopy(base['config']);c['training']['output']=str(root/'stopped')
            control=root/'control.json';control.write_text('{"action":"run"}')
            def event(kind,**values):
                if kind=='epoch' and values['epoch']==1:control.write_text('{"action":"stop"}')
            with patch.dict(os.environ,{'NLMS_CONTROL_FILE':str(control)}),patch('nlms_dfe.training.engine.event',event):
                self.assertIsNone(train(c))
            last=root/'stopped/last.pt';self.assertEqual(torch.load(last,weights_only=True)['epoch'],1)
            c['training'].update(resume=str(last),output=str(root/'resumed'))
            train(c)
            resumed=torch.load(root/'resumed/last.pt',weights_only=True)
            for name,value in base['state_dict'].items():torch.testing.assert_close(value,resumed['state_dict'][name],rtol=0,atol=0)
        finally:fixture.tearDown()

if __name__=='__main__':unittest.main()

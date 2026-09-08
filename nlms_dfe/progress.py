"""命令行和交互窗口共享的进度提示，不改变训练计算。"""
import json
import os
import time
from pathlib import Path


def event(kind, **values):
    if os.environ.get('NLMS_UI_EVENTS') == '1':
        print('@@NLMS@@'+json.dumps({'kind':kind,**values},ensure_ascii=False),flush=True)


class Progress:
    def __init__(self,stage,total,**context):
        self.stage,self.total,self.context=stage,int(total),context
        self.start=time.monotonic();self.last=-float('inf')

    def update(self,current,detail='',force=False):
        now=time.monotonic()
        if not force and current!=self.total and now-self.last<2:
            return
        elapsed=now-self.start
        remaining=elapsed*(self.total-current)/current if current else None
        percent=100*current/self.total if self.total else 0
        eta=f'{remaining:.0f}秒' if remaining is not None else '估算中'
        print(f'[{self.stage}] {current}/{self.total} ({percent:.1f}%) | 已用{elapsed:.0f}秒 | 剩余约{eta} | {detail}',flush=True)
        payload=dict(stage=self.stage,current=int(current),total=self.total,percent=percent,
                     elapsed_seconds=elapsed,remaining_seconds=remaining,detail=detail,**self.context)
        event('progress',**payload)
        path=os.environ.get('NLMS_PROGRESS_FILE')
        if path:
            Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        self.last=now

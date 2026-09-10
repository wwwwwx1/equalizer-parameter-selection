"""批次边界的协作控制；停止保留最近完整轮次供恢复。"""
import json
import os
import time
from pathlib import Path


def check_control():
    path=os.environ.get('NLMS_CONTROL_FILE')
    if not path:return
    announced=False
    while True:
        try:action=json.loads(Path(path).read_text(encoding='utf-8-sig')).get('action','run')
        except (OSError,ValueError):action='run'
        if action=='stop':raise KeyboardInterrupt
        if action!='pause':return
        if not announced:print('已暂停；继续后从当前批次边界运行。',flush=True);announced=True
        time.sleep(.3)

"""每2秒读取磁盘记录，不接触训练进程。"""
import csv
from pathlib import Path
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk


def show(root,folder):
    window=tk.Toplevel(root);window.title('实时训练曲线：'+str(folder))
    fig=Figure(figsize=(10,7));axes=fig.subplots(2,2)
    canvas=FigureCanvasTkAgg(fig,master=window);canvas.get_tk_widget().pack(fill='both',expand=True);NavigationToolbar2Tk(canvas,window)
    def read(name):
        path=folder/name
        if not path.exists():return []
        with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    def refresh():
        if not window.winfo_exists():return
        try:
            for ax in axes.flat:ax.clear()
            rows=read('training_history.csv')
            if rows:
                prefix='val' if 'val_selected_ber' in rows[0] else 'fit';x=[int(r['epoch']) for r in rows]
                # 新版使用每轮相同定义的完整目标，避免把 warmup/ramp 的数值跳变误读成未收敛。
                if all(r.get('monitor_fixed_objective','') and r.get(prefix+'_objective','') for r in rows):
                    axes[0,0].plot(x,[float(r['monitor_fixed_objective']) for r in rows],label='fixed train monitor')
                    axes[0,0].plot(x,[float(r[prefix+'_objective']) for r in rows],label=prefix)
                    axes[0,0].legend();axes[0,0].set_title('Fixed-definition objective (comparable)')
                else:
                    axes[0,0].plot(x,[float(r['train_loss']) for r in rows])
                    axes[0,0].set_title('Epoch training objective (stage-dependent)')
                for key in [prefix+'_selected_ber','fixed_ber','grid_best_ber']:axes[0,1].plot(x,[float(r[key]) for r in rows],label=key)
                axes[0,1].legend();axes[0,1].set_title('Validation BER' if prefix=='val' else 'Training fit BER')
                axes[1,1].plot(x,[float(r[prefix+'_regret']) for r in rows]);axes[1,1].set_title(prefix+' regret')
            batches=read('batch_history.csv')[-3000:]
            if batches:
                import numpy as np
                y=np.array([float(r['loss']) for r in batches]);axes[1,0].plot(y,alpha=.35,label='batch loss')
                if len(y)>=20:axes[1,0].plot(range(19,len(y)),np.convolve(y,np.ones(20)/20,mode='valid'),label='20-batch average')
                axes[1,0].legend();axes[1,0].set_title('Recent batch objective (stage-dependent; up to 3000)')
            for ax in axes.flat:ax.grid(alpha=.2)
            fig.tight_layout();canvas.draw_idle()
        except (OSError,ValueError,KeyError):pass
        window.after(2000,refresh)
    refresh()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=tk.Tk();root.withdraw();show(root,Path(a.output));root.mainloop()

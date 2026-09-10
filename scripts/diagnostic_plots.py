"""训练诊断面板；只展示真实记录，旧训练缺失的指标不补造。"""
import csv
from pathlib import Path
from matplotlib.figure import Figure


def figure(output):
    path=Path(output)/'diagnostic_history.csv'
    if not path.exists():raise ValueError('该训练尚未生成新版诊断记录；旧模型无法补出过去各轮的训练指标。')
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('等待第一轮诊断完成。')
    fig=Figure(figsize=(16,12),layout='constrained');axes=fig.subplots(4,3)
    x=[int(r['epoch']) for r in rows]
    def panel(ax,title,keys):
        for key in keys:
            if key in rows[0]:ax.plot(x,[float(r[key]) for r in rows],label=key)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=7)
    panel(axes[0,0],'BER: fixed train monitor / validation',['train_ber','val_ber','train_fixed','val_fixed'])
    panel(axes[0,1],'Mean regret / P90 regret',['train_regret','val_regret','train_p90_regret','val_p90_regret'])
    panel(axes[0,2],'Top1 / Top3 / Top5',['train_top1','val_top1','train_top3','val_top3','val_top5'])
    panel(axes[1,0],'Raw regression loss (eval mode)',['train_regression','val_regression'])
    panel(axes[1,1],'Raw soft-label CE (eval mode)',['train_soft','val_soft'])
    panel(axes[1,2],'Raw expected regret / ranking (eval)',['train_regret_loss','val_regret_loss','train_rank','val_rank'])
    panel(axes[2,0],'Weighted training contributions',['weighted_regression','weighted_soft','weighted_regret','weighted_rank','weighted_classification'])
    panel(axes[2,1],'Actual learning rate used',['lr_used'])
    panel(axes[2,2],'Gradient before clipping / clip fraction',['grad_norm_mean','clip_fraction'])
    panel(axes[3,0],'Fraction better / worse than fixed',['train_better','val_better','train_worse','val_worse'])
    panel(axes[3,1],'Epoch wall time / peak allocated GPU MB',['seconds','peak_gpu_mb'])
    panel(axes[3,2],'AMP skipped batches / auxiliary ramp',['amp_skipped_batches','weighted_auxiliary_weight'])
    fig.suptitle('Training diagnostics | fixed training subset; full validation | no test-set tuning')
    return fig


def export(output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=figure(output);FigureCanvasAgg(fig)
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/('diagnostics.'+ext),dpi=160)
    import shutil
    guide=Path(__file__).resolve().parents[1]/'docs/诊断图逐图解读.md'
    if guide.exists():shutil.copy2(guide,folder/guide.name)


def show(root,output):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
    window=tk.Toplevel(root);window.title('完整训练诊断（每10秒刷新；首轮后显示）')
    status=tk.StringVar();tk.Label(window,textvariable=status).pack()
    holder=tk.Frame(window);holder.pack(fill='both',expand=True)
    def refresh():
        if not window.winfo_exists():return
        try:
            fig=figure(output)
            for widget in holder.winfo_children():widget.destroy()
            canvas=FigureCanvasTkAgg(fig,master=holder);canvas.get_tk_widget().pack(fill='both',expand=True);canvas.draw()
            NavigationToolbar2Tk(canvas,holder)
            import json
            p=Path(output)/'diagnostic_status.json'
            status.set(json.loads(p.read_text(encoding='utf-8'))['message'] if p.exists() else '')
        except (OSError,ValueError,KeyError) as error:status.set(str(error))
        window.after(10000,refresh)
    refresh()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--live',action='store_true');args=p.parse_args()
    if args.live:
        import tkinter as tk
        root=tk.Tk();root.withdraw();show(root,Path(args.output));root.mainloop()
    else:export(args.output)

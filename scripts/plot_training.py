"""从实际日志绘制单次训练曲线，可在训练后单独运行。"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nlms_dfe.common import read_config,project_path


def plot(output):
    output=Path(output)
    with (output/'training_history.csv').open(encoding='utf-8-sig') as file:
        rows=list(csv.DictReader(file))
    if not rows:
        raise ValueError('训练日志没有数据。')
    prefix='val' if 'val_selected_ber' in rows[0] else 'fit'
    values=lambda key:np.array([float(r[key]) for r in rows])
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    epoch=values('epoch')
    axes[0,0].plot(epoch,values('train_loss'))
    axes[0,0].set(title='Training loss',xlabel='Epoch',ylabel='Loss')
    config=read_config(output/'config.json')
    warmup=config['loss']['warmup_epochs']
    axes[0,0].axvline(warmup+.5,color='gray',ls=':',label='Loss stage changes')
    axes[0,0].legend()
    for key,label,style in [('fit_selected_ber','Network','-'),('fixed_ber','Fixed','--'),('grid_best_ber','Grid minimum',':')]:
        axes[0,1].plot(epoch,values(key.replace('fit_',prefix+'_'))*100,style,label=label)
    axes[0,1].set(title=('Validation' if prefix=='val' else 'Training')+' MeanCodedBER',xlabel='Epoch',ylabel='BER (%)')
    axes[0,1].legend()
    axes[1,0].plot(epoch,values(prefix+'_top1')*100,label='Top-1')
    axes[1,0].plot(epoch,values(prefix+'_top3')*100,label='Top-3')
    axes[1,0].set(title='Tie-aware grid-best recovery',xlabel='Epoch',ylabel='Hit rate (%)',ylim=(0,100))
    axes[1,0].legend()
    axes[1,1].plot(epoch,values(prefix+'_regret'))
    axes[1,1].set(title=('Validation' if prefix=='val' else 'Training')+' regret',xlabel='Epoch',ylabel='Selected BER - grid minimum')
    fig.suptitle('MeanCodedBER | '+('Validation curves; test evaluated separately' if prefix=='val' else 'Training fit; no held-out evaluation'),fontsize=13)
    folder=output/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(folder/f'training_curves.{ext}',dpi=300)
    plt.close(fig)
    if (output/'diagnostic_history.csv').exists():
        from scripts.diagnostic_plots import export
        export(output)
    print('训练图：',folder)


if __name__=='__main__':
    plot(project_path(read_config('configs/train.json')['training']['output']))

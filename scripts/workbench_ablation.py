"""工作台消融页面：任务状态、曲线、结果与模型切换。"""
import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from scripts.ablation_variants import VARIANTS
from nlms_dfe.common import ROOT,read_config


class AblationPage:
    def __init__(self,app):
        self.app=app;self.folder=None;self.tasks=[]
        p=self.page=ttk.Frame(app.tabs,padding=10);app.tabs.add(p,text='消融与多模型');p.columnconfigure(0,weight=1)
        ttk.Label(p,text='沿用“模型训练”页的数据、裁剪、轮数和批次；模型配置页可修改共享结构。Ctrl/Shift可多选变体。').grid(row=0,column=0,sticky='w')
        self.names=list(VARIANTS);self.options=tk.Listbox(p,selectmode='extended',exportselection=False,height=5)
        for name in self.names:self.options.insert('end',VARIANTS[name]['label']+'  |  '+name)
        self.options.selection_set(0);self.options.grid(row=1,column=0,sticky='ew')
        self.options.bind('<<ListboxSelect>>',self.describe)
        self.description=tk.StringVar();ttk.Label(p,textvariable=self.description,wraplength=980).grid(row=2,column=0,sticky='w')
        settings=ttk.Frame(p);settings.grid(row=3,column=0,sticky='ew',pady=4)
        self.mode=tk.StringVar(value='组外比较');self.parallel=tk.StringVar(value='1');self.seeds=tk.StringVar(value='42');self.split_seed=tk.StringVar(value='20260909')
        ttk.Combobox(settings,textvariable=self.mode,values=['组外比较','全量训练'],width=12,state='readonly').pack(side='left')
        for label,var,width in [('并发数',self.parallel,4),('训练种子（逗号分隔）',self.seeds,15),('划分种子',self.split_seed,12)]:
            ttk.Label(settings,text=label).pack(side='left',padx=5);ttk.Entry(settings,textvariable=var,width=width).pack(side='left')
        ttk.Button(settings,text='选择历史实验',command=self.load).pack(side='left',padx=8)
        self.output=tk.StringVar(value=str(ROOT/'outputs/model_comparisons'))
        row=ttk.Frame(p);row.grid(row=4,column=0,sticky='ew');row.columnconfigure(1,weight=1)
        app.entry(row,0,'结果根目录',self.output,'dir')
        self.table=ttk.Treeview(p,columns=('state','epoch','loss','ber','fixed','regret','top1'),height=6)
        self.table.heading('#0',text='模型 / 种子');self.table.column('#0',width=250)
        for key,label in [('state','状态 / 口径'),('epoch','轮数'),('loss','Loss'),('ber','BER %'),('fixed','固定 BER %'),('regret','Regret'),('top1','Top1 %')]:
            self.table.heading(key,text=label);self.table.column(key,width=92)
        self.table.grid(row=5,column=0,sticky='nsew');p.rowconfigure(5,weight=1)
        bar=ttk.Frame(p);bar.grid(row=6,column=0,sticky='w')
        for label,fn in [('对比所选曲线',self.curves),('最终BER对比图',self.compare),('打开所选日志',self.log),('使用所选模型预测',self.use),('打开实验目录',self.open)]:ttk.Button(bar,text=label,command=fn).pack(side='left',padx=4)
        self.hint=tk.StringVar(value='组外模式按来源组约70/20/10划分；验证选轮次，测试仅在训练结束评估。多次查看测试后仍需独立新数据最终验证。')
        ttk.Label(p,textvariable=self.hint,wraplength=980).grid(row=7,column=0,sticky='w')
        old_job=app.job
        def job():return self.job() if app.tabs.select()==str(p) else old_job()
        app.job=job
        app.root.after(1000,self.refresh)

    def describe(self,event=None):
        ix=self.options.curselection()
        if ix:self.description.set(VARIANTS[self.names[ix[-1]]]['tests'])

    def job(self):
        from scripts.workbench_extensions import validate_model_config
        a=self.app;c=copy.deepcopy(a.config)
        c['data'].update(manifest=a.manifest.get(),candidates=a.candidates.get(),cache_features=a.cache.get())
        c['data']['crop'].update(main_path_matlab=int(a.train_center.get()),already_cropped=a.train_cropped.get(),allow_short=a.train_short.get())
        c['model']['layers']=int(a.layers.get());c['training'].update(epochs=int(a.epochs.get()),batch_size=int(a.batch.get()),target_metric='MeanCodedBER')
        validate_model_config(c)
        for k in ('manifest','candidates'):
            if not Path(c['data'][k]).is_file():raise ValueError('请在训练页选择新电脑的索引与候选参数文件。')
        names=[self.names[i] for i in self.options.curselection()]
        if not names:raise ValueError('至少选择一个模型。')
        seeds=list(dict.fromkeys(int(x.strip()) for x in self.seeds.get().replace('，',',').split(',')))
        concurrency=int(self.parallel.get()); split=int(self.split_seed.get())
        if not 1<=concurrency<=8 or any(not 0<=x<2**32 for x in seeds+[split]):raise ValueError('并发数为1至8，种子为0至4294967295。')
        self.folder=a.new_path(self.output.get(),datetime.now().strftime('%Y%m%d_%H%M%S_%f'));self.tasks=[]
        return dict(action='ablation',split_ratio=[.7,.2,.1],config=c,output=str(self.folder),variants=names,seeds=seeds,split_seed=split,concurrency=concurrency,mode='holdout' if self.mode.get()=='组外比较' else 'all_data')

    def load(self):
        path=filedialog.askdirectory(title='选择含tasks.json的实验目录')
        if path:self.folder=Path(path);self.tasks=[]

    def refresh(self):
        try:
            if self.folder and (self.folder/'tasks.json').exists():
                self.tasks=read_config(self.folder/'tasks.json')
                for task in self.tasks:
                    key=task['id'];folder=Path(task['config']['training']['output'])
                    # 历史实验目录整体移动后仍可查看。
                    if not folder.exists():folder=self.folder/key
                    task['_folder']=folder
                    status_path=self.folder/(key+'.status.json');state=read_config(status_path)['status'] if status_path.exists() else '等待'
                    epoch=loss=ber=fixed=regret=top1=''
                    history=folder/'training_history.csv'
                    if history.exists():
                        with history.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                        if rows:
                            r=rows[-1];prefix='val' if 'val_selected_ber' in r else 'fit';epoch=r['epoch'];loss=f"{float(r['train_loss']):.4g}"
                            ber=f"{100*float(r[prefix+'_selected_ber']):.4f}";fixed=f"{100*float(r['fixed_ber']):.4f}";regret=f"{float(r[prefix+'_regret']):.5f}";top1=f"{100*float(r[prefix+'_top1']):.2f}";state+=' / '+('验证' if prefix=='val' else '拟合')
                    result=folder/'result.json'
                    if result.exists():
                        r=read_config(result);m=r['metrics'];state='完成 / '+r['scope'];ber=f"{100*m['selected_ber_macro']:.4f}";fixed=f"{100*m['fixed_ber_macro']:.4f}";regret=f"{m['mean_regret']:.5f}";top1=f"{100*m['top1_tie_aware_accuracy']:.2f}"
                    values=(state,epoch,loss,ber,fixed,regret,top1)
                    if self.table.exists(key):self.table.item(key,values=values)
                    else:self.table.insert('','end',iid=key,text=f"{task['label']} / {task['seed']}",values=values)
                ids={t['id'] for t in self.tasks}
                for key in self.table.get_children():
                    if key not in ids:self.table.delete(key)
        except (OSError,ValueError,KeyError):pass # 文件写入中，下次刷新读取完整快照。
        self.app.root.after(1500,self.refresh)

    def selected(self):
        ids=self.table.selection();return [t for t in self.tasks if t['id'] in ids]
    def open(self):
        if self.folder and self.folder.exists():os.startfile(self.folder)
    def log(self):
        for t in self.selected():
            path=self.folder/(t['id']+'.log')
            if path.exists():os.startfile(path)
    def use(self):
        selected=self.selected()
        if len(selected)!=1:messagebox.showinfo('选择模型','请只选一个已完成模型。');return
        path=selected[0]['_folder']/'result.json'
        if not path.exists():messagebox.showinfo('模型未完成','等待训练和评估完成。');return
        result=read_config(path);checkpoint=Path(result['checkpoint'])
        if not checkpoint.exists():checkpoint=path.parent/checkpoint.name
        self.app.checkpoint.set(str(checkpoint));self.app.tabs.select(self.app.pages[2])
    def compare(self):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
            tasks=self.selected() or self.tasks
            results=[(t,read_config(t['_folder']/'result.json')) for t in tasks if (t['_folder']/'result.json').exists()]
            if not results:raise ValueError('还没有已完成的模型。')
            fig=Figure(figsize=(10,5));ax=fig.subplots();x=list(range(len(results)))
            ax.bar(x,[r['metrics']['selected_ber_macro']*100 for t,r in results],label='Model')
            ax.plot(x,[r['metrics']['fixed_ber_macro']*100 for t,r in results],'r--o',label='Train-selected fixed')
            ax.set_xticks(x,[t['id'] for t,r in results],rotation=25,ha='right');ax.set_ylabel('MeanCodedBER (%)');ax.set_title('Held-out test' if results[0][1]['scope']=='组外测试' else 'Training fit only');ax.legend();fig.tight_layout()
            window=tk.Toplevel(self.app.root);window.title('已完成模型与固定参数比较')
            canvas=FigureCanvasTkAgg(fig,master=window);canvas.draw();canvas.get_tk_widget().pack(fill='both',expand=True);NavigationToolbar2Tk(canvas,window)
            fig.savefig(self.folder/'selected_final_comparison.png',dpi=150)
        except Exception as e:messagebox.showerror('无法比较',str(e))

    def curves(self):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
            tasks=self.selected() or self.tasks
            fig=Figure(figsize=(10,6));axes=fig.subplots(2,2)
            for t in tasks:
                path=t['_folder']/'training_history.csv'
                if not path.exists():continue
                with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                if not rows:continue
                prefix='val' if 'val_selected_ber' in rows[0] else 'fit'
                for ax,key in zip(axes.flat,['train_loss',prefix+'_selected_ber',prefix+'_regret',prefix+'_top1']):
                    ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=t['id'])
            for ax,title in zip(axes.flat,['Training loss','Validation BER / Fit BER','Validation regret / Fit regret','Validation Top1 / Fit Top1']):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=6);fig.tight_layout()
            window=tk.Toplevel(self.app.root);window.title('多模型训练曲线（测试结果见主表）')
            canvas=FigureCanvasTkAgg(fig,master=window);canvas.draw();canvas.get_tk_widget().pack(fill='both',expand=True);NavigationToolbar2Tk(canvas,window)
            if self.folder:fig.savefig(self.folder/'selected_training_comparison.png',dpi=150)
        except Exception as e:messagebox.showerror('曲线读取失败',str(e))


def install(app):app.ablation_page=AblationPage(app)

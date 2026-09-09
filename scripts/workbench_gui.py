"""统一操作窗口：数据准备、训练、单信道和批量预测。仅使用标准库构建界面。"""
import copy
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nlms_dfe.common import active_model,project_path,read_config,write_json


def console_python():
    configured=ROOT/'python_path.txt'
    if configured.exists() and configured.read_text(encoding='utf-8-sig').strip():
        return configured.read_text(encoding='utf-8-sig').strip()
    path=Path(sys.executable)
    if path.name.lower()=='pythonw.exe' and path.with_name('python.exe').exists():
        path=path.with_name('python.exe')
    return str(path)


class Workbench:
    def __init__(self,root):
        self.root=root;self.events=queue.Queue();self.running=False
        self.process=None;self.last_output=None;self.last_result=None
        self.started=None;self.phase='等待开始';self.progress_text=''
        self.current_round=''
        self.progress_current=0;self.progress_total=0
        root.title('信道参数选择工作台 — MeanCodedBER')
        root.geometry('1100x890');root.minsize(950,780)
        root.protocol('WM_DELETE_WINDOW',self.close)
        style=ttk.Style(root);style.theme_use('clam');style.configure('.',font=('Microsoft YaHei UI',10))
        frame=ttk.Frame(root,padding=16);frame.pack(fill='both',expand=True)
        frame.columnconfigure(1,weight=1);frame.rowconfigure(5,weight=1)
        self.python=tk.StringVar(value=console_python())
        self.checkpoint=tk.StringVar(value=str(project_path(active_model()['checkpoint'])))
        self.entry(frame,0,'Python解释器',self.python,'file')
        self.entry(frame,1,'预测使用的模型',self.checkpoint,'file')
        self.tabs=ttk.Notebook(frame);self.tabs.grid(row=2,column=0,columnspan=3,sticky='nsew',pady=12)
        self.pages=[]
        for name in ['数据准备','模型训练','单信道预测','批量预测']:
            page=ttk.Frame(self.tabs,padding=12);page.columnconfigure(1,weight=1)
            self.tabs.add(page,text=name);self.pages.append(page)
        default='configs/train_window.json' if (ROOT/'configs/train_window.json').exists() else 'configs/train.json'
        source='configs/dataset_window.json' if (ROOT/'configs/dataset_window.json').exists() else 'configs/dataset.json'
        self.source_template=read_config(source)
        self.config_path=tk.StringVar(value=str(project_path(default)))
        self.config=read_config(default)
        self.build_prepare();self.build_train();self.build_single();self.build_batch()
        self.tabs.select(2 if Path(self.checkpoint.get()).is_file() else 0)
        buttons=ttk.Frame(frame);buttons.grid(row=3,column=0,columnspan=3,sticky='ew')
        self.run_button=ttk.Button(buttons,text='开始当前页任务',command=self.start);self.run_button.pack(side='left')
        self.open_button=ttk.Button(buttons,text='打开本次结果',command=self.open_output,state='disabled');self.open_button.pack(side='left',padx=12)
        ttk.Button(buttons,text='打开日志目录',command=self.open_logs).pack(side='left')
        self.status=tk.StringVar(value='选择上方模式，填写设置，再点击开始。新电脑可选择已有Python，无需在线安装。')
        panel=ttk.Frame(frame);panel.grid(row=4,column=0,columnspan=3,sticky='ew',pady=10)
        panel.columnconfigure(0,weight=1)
        ttk.Label(panel,textvariable=self.status,wraplength=1020).grid(row=0,column=0,sticky='w')
        self.bar=ttk.Progressbar(panel,maximum=100);self.bar.grid(row=1,column=0,sticky='ew',pady=(8,0))
        logframe=ttk.LabelFrame(frame,text='实时日志（同时保存到文件）',padding=6)
        logframe.grid(row=5,column=0,columnspan=3,sticky='nsew');logframe.rowconfigure(0,weight=1);logframe.columnconfigure(0,weight=1)
        self.log=tk.Text(logframe,height=13,wrap='word',font=('Microsoft YaHei UI',9),state='disabled')
        self.log.grid(row=0,column=0,sticky='nsew')
        scrollbar=ttk.Scrollbar(logframe,command=self.log.yview);scrollbar.grid(row=0,column=1,sticky='ns');self.log.configure(yscrollcommand=scrollbar.set)
        ttk.Label(frame,text='训练页按来源组7:2:1划分；val是验证指标，训练结束评估测试集。预测只推荐参数，不运行均衡器或计算新信道BER。').grid(row=6,column=0,columnspan=3,sticky='w',pady=(8,0))
        root.after(100,self.poll)
        from scripts.workbench_extensions import install
        install(self)
        from scripts.workbench_ablation import install as install_ablation
        install_ablation(self)
        root.geometry(f"1100x{min(960,max(780,root.winfo_screenheight()-100))}")

    def entry(self,parent,row,label,var,kind=None):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',padx=(0,10),pady=4)
        ttk.Entry(parent,textvariable=var).grid(row=row,column=1,sticky='ew',pady=4)
        if kind:
            ttk.Button(parent,text='选择…',command=lambda:self.browse(var,kind)).grid(row=row,column=2,padx=(8,0))

    def browse(self,var,kind):
        value=filedialog.askdirectory() if kind=='dir' else filedialog.askopenfilename()
        if value:var.set(value)

    def build_prepare(self):
        p=self.pages[0];s=self.source_template;c=self.config['data']['crop']
        self.channel_dir=tk.StringVar(value=s['channel_dir']);self.result_dir=tk.StringVar(value=s['result_dir'])
        self.prepare_output=tk.StringVar(value=str(ROOT/'data/window_prepared'))
        self.pre_center=tk.StringVar(value=str(c['main_path_matlab']));self.pre_short=tk.BooleanVar(value=c.get('allow_short',True))
        self.pre_cropped=tk.BooleanVar(value=False)
        self.entry(p,0,'原始信道目录',self.channel_dir,'dir');self.entry(p,1,'网格结果目录',self.result_dir,'dir')
        self.entry(p,2,'准备结果根目录',self.prepare_output,'dir');self.entry(p,3,'原始主径列号',self.pre_center)
        ttk.Checkbutton(p,text='原始数据不足4000行时使用实际末尾',variable=self.pre_short).grid(row=4,column=0,columnspan=3,sticky='w')
        ttk.Checkbutton(p,text='原始文件已裁好（不再裁剪）',variable=self.pre_cropped).grid(row=5,column=0,columnspan=3,sticky='w')
        ttk.Label(p,text='按模板字段读取h_vary与MeanCodedBER；自动建立索引。每次生成独立目录，完成后自动填写训练页。').grid(row=6,column=0,columnspan=3,sticky='w',pady=6)

    def build_train(self):
        p=self.pages[1]
        self.entry(p,0,'训练配置文件',self.config_path,'file')
        self.manifest=tk.StringVar();self.candidates=tk.StringVar();self.train_output=tk.StringVar(value=str(ROOT/'outputs/window_training'))
        self.entry(p,1,'数据索引CSV',self.manifest,'file');self.entry(p,2,'候选参数CSV',self.candidates,'file')
        self.entry(p,3,'训练输出根目录',self.train_output,'dir')
        settings=ttk.Frame(p);settings.grid(row=4,column=0,columnspan=3,sticky='w',pady=5)
        self.epochs=tk.StringVar();self.batch=tk.StringVar();self.layers=tk.StringVar();self.train_center=tk.StringVar()
        for label,var in [('轮数',self.epochs),('每批样本',self.batch),('时间网络层数',self.layers),('原始主径',self.train_center)]:
            ttk.Label(settings,text=label).pack(side='left',padx=(0,6));ttk.Entry(settings,textvariable=var,width=6).pack(side='left',padx=(0,15))
        self.train_cropped=tk.BooleanVar();self.train_short=tk.BooleanVar();self.cache=tk.BooleanVar()
        options=ttk.Frame(p);options.grid(row=5,column=0,columnspan=3,sticky='w')
        for label,var in [('索引指向已裁剪信道',self.train_cropped),('允许短时间记录',self.train_short),('内存缓存特征（仅小数据集）',self.cache)]:
            ttk.Checkbutton(options,text=label,variable=var).pack(side='left',padx=(0,12))
        ttk.Button(p,text='读取配置并更新本页',command=self.load_train_config).grid(row=6,column=0,sticky='w',pady=4)
        ttk.Label(p,text='数据已准备好时，可直接在本页训练；自动按来源组7:2:1划分，显示扫描、训练和验证进度。').grid(row=6,column=1,columnspan=2,sticky='w')
        self.assign_config(self.config)

    def assign_config(self,c):
        self.config=copy.deepcopy(c);crop=c['data']['crop']
        self.manifest.set(str(project_path(c['data']['manifest'])));self.candidates.set(str(project_path(c['data']['candidates'])))
        self.epochs.set(str(c['training']['epochs']));self.batch.set(str(c['training']['batch_size']));self.layers.set(str(c['model']['layers']))
        self.train_center.set(str(crop['main_path_matlab']));self.train_cropped.set(crop['already_cropped'])
        self.train_short.set(crop.get('allow_short',True));self.cache.set(c['data'].get('cache_features',False))

    def load_train_config(self):
        try:self.assign_config(read_config(self.config_path.get()))
        except Exception as error:messagebox.showerror('无法读取配置',str(error))

    def prediction_options(self,p,row,prefix):
        center=tk.StringVar(value=str(self.config['data']['crop']['main_path_matlab']))
        cropped=tk.BooleanVar(value=False);short=tk.BooleanVar(value=True)
        options=ttk.Frame(p);options.grid(row=row,column=0,columnspan=3,sticky='w',pady=6)
        ttk.Label(options,text='原始主径列号').pack(side='left');ttk.Entry(options,textvariable=center,width=7).pack(side='left',padx=8)
        ttk.Checkbutton(options,text='信道已裁好',variable=cropped).pack(side='left',padx=10)
        ttk.Checkbutton(options,text='短信道用实际末尾',variable=short).pack(side='left',padx=10)
        setattr(self,prefix+'_center',center);setattr(self,prefix+'_cropped',cropped);setattr(self,prefix+'_short',short)

    def build_single(self):
        p=self.pages[2];self.single_path=tk.StringVar();self.single_snr=tk.StringVar()
        self.single_output=tk.StringVar(value=str(ROOT/'outputs/single_predictions'))
        self.entry(p,0,'信道MAT文件',self.single_path,'file');self.entry(p,1,'SNR（dB，可为负数）',self.single_snr)
        self.entry(p,2,'结果保存根目录',self.single_output,'dir');self.prediction_options(p,3,'single')
        self.answer=tk.StringVar(value='推荐参数会在这里显示。')
        ttk.Label(p,textvariable=self.answer,font=('Microsoft YaHei UI',15,'bold')).grid(row=4,column=0,columnspan=3,sticky='w',pady=8)
        self.copy_button=ttk.Button(p,text='复制MATLAB参数',command=self.copy_parameters,state='disabled');self.copy_button.grid(row=5,column=0,sticky='w')

    def build_batch(self):
        p=self.pages[3];self.batch_dir=tk.StringVar(value=self.source_template.get('channel_dir',''))
        batch_config=ROOT/'批量信道预测/config.json'
        if batch_config.exists():self.batch_dir.set(read_config(batch_config)['channel_dir'])
        self.batch_output=tk.StringVar(value=str(ROOT/'outputs/batch_predictions'))
        self.entry(p,0,'信道文件夹',self.batch_dir,'dir');self.entry(p,1,'结果保存根目录',self.batch_output,'dir')
        settings=ttk.Frame(p);settings.grid(row=2,column=0,columnspan=3,sticky='w')
        self.low=tk.StringVar(value='-5');self.high=tk.StringVar(value='5');self.seed=tk.StringVar()
        for label,var in [('随机SNR下限',self.low),('上限',self.high),('随机种子（可空）',self.seed)]:
            ttk.Label(settings,text=label).pack(side='left');ttk.Entry(settings,textvariable=var,width=12).pack(side='left',padx=10)
        self.prediction_options(p,3,'batch')
        self.recursive=tk.BooleanVar(value=False)
        ttk.Checkbutton(p,text='包含子文件夹',variable=self.recursive).grid(row=4,column=0,sticky='w')
        ttk.Label(p,text='每信道随机指定一个SNR，保留3位小数。汇总MAT包含信道名、SNR、μ、N1、N2。').grid(row=5,column=0,columnspan=3,sticky='w',pady=8)

    def job(self):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        mode=self.tabs.index(self.tabs.select())
        c=copy.deepcopy(self.config)
        if mode==0:
            source=copy.deepcopy(self.source_template)
            source.update(channel_dir=self.channel_dir.get().strip(),result_dir=self.result_dir.get().strip(),
                          output_dir=str(self.new_path(self.prepare_output.get(),stamp)))
            for key in ['channel_dir','result_dir']:
                if not Path(source[key]).is_dir() or not source[key]:raise ValueError('请选择原始信道目录和结果目录。')
            c['data']['crop'].update(main_path_matlab=self.positive(self.pre_center.get()),already_cropped=self.pre_cropped.get(),allow_short=self.pre_short.get())
            return dict(action='prepare',source=source,config=c)
        if mode==1:
            if not Path(self.manifest.get()).is_file() or not Path(self.candidates.get()).is_file():raise ValueError('请先准备数据，或选择有效索引与候选CSV。')
            c['data'].update(manifest=self.manifest.get(),candidates=self.candidates.get(),cache_features=self.cache.get())
            c['data']['crop'].update(main_path_matlab=self.positive(self.train_center.get()),already_cropped=self.train_cropped.get(),allow_short=self.train_short.get())
            c['model']['layers']=self.positive(self.layers.get())
            c['training'].update(epochs=self.positive(self.epochs.get()),batch_size=self.positive(self.batch.get()),
                                 mode='holdout',target_metric='MeanCodedBER',output=str(self.new_path(self.train_output.get(),stamp)))
            if c['data']['fields']['ber']!='mean_coded_ber':raise ValueError('当前工作台只训练MeanCodedBER，请核对配置的标签字段。')
            return dict(action='train',config=c,split_ratio=[.7,.2,.1])
        checkpoint=self.checkpoint.get().strip()
        if not Path(checkpoint).is_file():raise ValueError('请选择已训练好的模型；新电脑需先训练。')
        if mode==2:
            channel=self.single_path.get().strip()
            if not Path(channel).is_file():raise ValueError('请选择原始信道MAT。')
            snr=float(self.single_snr.get())
            if not math.isfinite(snr):raise ValueError('SNR必须是有限数字。')
            return dict(action='single',checkpoint=checkpoint,channel=channel,snr=snr,field='h_vary',
                        center=self.positive(self.single_center.get()),cropped=self.single_cropped.get(),allow_short=self.single_short.get(),
                        output=str(self.new_path(self.single_output.get(),stamp)))
        source=self.batch_dir.get().strip();low=float(self.low.get());high=float(self.high.get())
        if not source or not Path(source).is_dir():raise ValueError('请选择信道文件夹。')
        if not math.isfinite(low) or not math.isfinite(high) or low>=high:raise ValueError('请填写有限且下限小于上限的SNR范围。')
        seed=int(self.seed.get()) if self.seed.get().strip() else None
        if seed is not None and not 0<=seed<2**32:raise ValueError('随机种子应为0至4294967295的整数，或留空。')
        output=self.batch_output.get().strip()
        if not output:raise ValueError('请选择结果根目录。')
        return dict(action='batch',batch=dict(channel_dir=source,output_dir=str(project_path(output)),checkpoint=checkpoint,
                    file_pattern='*.mat',recursive=self.recursive.get(),channel_field='h_vary',
                    main_path_matlab=self.positive(self.batch_center.get()),already_cropped=self.batch_cropped.get(),allow_short=self.batch_short.get(),
                    snr_min_db=low,snr_max_db=high,snr_decimal_places=3,random_seed=seed,device='cpu'))

    @staticmethod
    def positive(value):
        number=int(value)
        if number<1:raise ValueError('轮数、批次、层数和主径列号必须为正整数。')
        return number

    @staticmethod
    def new_path(value,stamp):
        if not value.strip():raise ValueError('请填写输出根目录。')
        return project_path(value.strip())/stamp

    def start(self):
        if self.running:return
        try:
            python=self.python.get().strip()
            if not Path(python).is_file():raise ValueError('请选择已有环境里的python.exe。')
            job=self.job()
            logs=ROOT/'outputs/ui_logs';logs.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.job_path=logs/f'{stamp}.json';write_json(self.job_path,job)
            self.log_path=logs/f'{stamp}.log'
            self.progress_path=logs/f'{stamp}.progress.json'
            (ROOT/'python_path.txt').write_text(python,encoding='utf-8')
        except Exception as error:messagebox.showerror('请检查设置',str(error));return
        self.running=True;self.started=time.monotonic();self.last_result=None;self.progress_text=''
        self.current_round=''
        self.phase='启动后台进程';self.bar['value']=0;self.run_button.configure(state='disabled');self.open_button.configure(state='disabled')
        self.log.configure(state='normal');self.log.delete('1.0','end');self.log.configure(state='disabled')
        threading.Thread(target=self.worker,args=(python,self.job_path,self.log_path,self.progress_path),daemon=True).start()

    def worker(self,python,job_path,log_path,progress_path):
        try:
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',NLMS_UI_EVENTS='1',NLMS_PROGRESS_FILE=str(progress_path))
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            with log_path.open('w',encoding='utf-8') as file:
                process=subprocess.Popen([python,'-X','utf8','-u','-m','scripts.workbench_job',str(job_path)],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,creationflags=flags)
                self.process=process
                for line in process.stdout:
                    file.write(line);file.flush()
                    if line.startswith('@@NLMS@@'):
                        try:self.events.put(json.loads(line[len('@@NLMS@@'):]))
                        except json.JSONDecodeError:self.events.put({'kind':'log','text':line})
                    else:self.events.put({'kind':'log','text':line})
                code=process.wait()
            self.events.put({'kind':'exit','code':code})
        except Exception as error:self.events.put({'kind':'failure','text':str(error)})

    def append_log(self,text):
        self.log.configure(state='normal');self.log.insert('end',text)
        if int(self.log.index('end-1c').split('.')[0])>2500:self.log.delete('1.0','501.0')
        self.log.see('end');self.log.configure(state='disabled')

    def poll(self):
        for _ in range(250):
            try:item=self.events.get_nowait()
            except queue.Empty:break
            kind=item['kind']
            if kind=='log':self.append_log(item['text'])
            elif kind=='progress':
                self.phase=item['stage'];self.bar['value']=item['percent']
                if 'epoch' in item:self.current_round=f"第{item['epoch']}/{item['epochs']}轮 | "
                eta=item.get('remaining_seconds');remaining=f'；本阶段剩余约{eta:.0f}秒' if eta is not None else ''
                self.progress_text=f"{item['current']}/{item['total']} ({item['percent']:.1f}%){remaining} | {item.get('detail','')}"
            elif kind=='epoch':self.append_log(f"第{item['epoch']}/{item['epochs']}轮完成，{item['scope']} BER={item['ber']:.6g}\n")
            elif kind=='prediction':
                row=item['row']
                if row['success']:self.append_log(f"{row['channel_name']} | SNR={row['snr_db']} | μ={row['mu']:.7g}, N1={row['N1']}, N2={row['N2']}\n")
            elif kind=='result':self.accept_result(item)
            elif kind in ('exit','failure'):
                self.running=False;self.run_button.configure(state='normal');self.process=None
                if kind=='failure' or item['code']!=0:
                    self.status.set('任务失败；请查看下方报错或打开日志目录。')
                    if kind=='failure':self.append_log(item['text']+'\n')
                elif self.last_result:
                    self.status.set(self.last_result['message']);self.bar['value']=100
                else:self.status.set('进程结束，但未收到完成结果，请检查日志。')
        if self.running:
            elapsed=time.monotonic()-self.started
            self.status.set(f'{self.phase} | {self.current_round}总耗时 {elapsed:.0f}秒 | {self.progress_text}')
        self.root.after(100,self.poll)

    def accept_result(self,item):
        self.last_result=item;self.last_output=Path(item['output']);self.open_button.configure(state='normal')
        self.append_log(item['message']+'\n')
        if item['action']=='prepare':
            self.config_path.set(item['config_path']);self.assign_config(read_config(item['config_path']))
        elif item['action']=='train':self.checkpoint.set(item['checkpoint'])
        elif item['action']=='single':
            p=item['prediction'];self.answer.set(f"μ = {p['mu']:.7g}    N1 = {p['N1']}    N2 = {p['N2']}")
            self.matlab=f"mu = {p['mu']:.7g};\nN1 = {p['N1']};\nN2 = {p['N2']};";self.copy_button.configure(state='normal')

    def copy_parameters(self):self.root.clipboard_clear();self.root.clipboard_append(self.matlab)
    def open_output(self):
        if self.last_output:os.startfile(self.last_output)
    def open_logs(self):
        folder=ROOT/'outputs/ui_logs';folder.mkdir(parents=True,exist_ok=True);os.startfile(folder)
    def close(self):
        if self.running:messagebox.showinfo('任务运行中','请等待任务完成再关闭，避免丢失尚未保存的模型或MAT。');return
        self.root.destroy()


if __name__=='__main__':
    root=tk.Tk();Workbench(root);root.mainloop()

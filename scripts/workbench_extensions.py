"""模型配置编辑器、MATLAB仿真和结果分析页面。"""
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox,simpledialog
from nlms_dfe.common import ROOT,read_config,write_json,project_path


DESCRIPTIONS={'layers':'时间Transformer层数','d_model':'每个时间token的特征宽度','heads':'注意力头数，需要整除d_model',
              'ffn_dim':'Transformer前馈隐藏维度','delay_bins':'时延池化分箱数','candidate_dim':'候选参数表示维度',
              'dropout':'丢弃比例，范围0至1','temporal_pool':'每多少行时间特征合并一次','learning_rate':'初始学习率',
              'weight_decay':'AdamW权重衰减','batch_size':'每批信道数','epochs':'训练轮次',
              'score_weight':'性能差距回归权重','soft_weight':'软标签权重','regret_weight':'期望regret权重',
              'rank_weight':'排序损失权重','temperature':'软标签温度','warmup_epochs':'仅回归预热轮数',
              'main_path_matlab':'原始信道主径的一基列号','cache_features':'大数据集建议关闭',
              'features':'ri / abs / ri_abs / ri_abs_phase'}


def validate_model_config(c):
    m,t,d=c['model'],c['training'],c['data']
    for value in [m[k] for k in ('layers','d_model','heads','ffn_dim','delay_bins','candidate_dim')]+[t['epochs'],t['batch_size'],d['temporal_pool']]:
        if not isinstance(value,int) or isinstance(value,bool) or value<1:raise ValueError('层数、维度、头数、轮数、批次和池化长度需要正整数。')
    if m['d_model']%2 or m['d_model']%m['heads']:raise ValueError('d_model需要是偶数，且能被heads整除。')
    if not 0<=m['dropout']<1:raise ValueError('dropout应在[0,1)。')
    if m['kind'] not in ('scorer','classifier'):raise ValueError('model.kind应为scorer或classifier。')
    if m['snr_conditioning'] not in ('film','concat'):raise ValueError('条件化方式应为film或concat。')
    if d['features'] not in ('ri','abs','ri_abs','ri_abs_phase'):raise ValueError('未知信道特征组合。')
    if t['learning_rate']<=0 or c['loss']['temperature']<=0 or c['loss']['scale_floor']<=0:raise ValueError('学习率、温度和scale_floor需大于0。')
    if c['data']['fields']['ber']!='mean_coded_ber' or t['target_metric']!='MeanCodedBER':raise ValueError('当前工作台训练目标为MeanCodedBER。')
    json.dumps(c,allow_nan=False)


class Extensions:
    def __init__(self,app):
        self.app=app;self.edit_config=copy.deepcopy(app.config);self.paths={};self.sim={}
        self.mcfg=read_config('configs/matlab_workbench.json')
        self.add_editor();self.add_simulation('prediction','预测参数MATLAB仿真');self.add_simulation('grid','294组MATLAB仿真');self.add_analysis()
        old_job=app.job
        def job():
            page=app.tabs.select()
            if page==str(self.editor):raise ValueError('此页用于编辑配置，请点击“应用到训练页”，再到模型训练页启动。')
            for mode,values in self.sim.items():
                if page==str(values['page']):return self.matlab_job(mode)
            if page==str(self.analysis):return self.analysis_job()
            return old_job()
        app.job=job
        original=app.accept_result
        def accept(item):
            original(item)
            if item['action']=='batch':
                mat=item['summary']['mat_path']
                for values in self.sim.values():values['input'].set(mat)
                self.analysis_prediction.set(mat)
            elif item['action']=='matlab_prediction':self.analysis_sim.set(item['output_mat'])
            elif item['action']=='matlab_grid':self.analysis_grid.set(item['output_mat'])
        app.accept_result=accept

    def page(self,title):
        p=ttk.Frame(self.app.tabs,padding=10);p.columnconfigure(1,weight=1)
        self.app.tabs.add(p,text=title);return p

    def add_editor(self):
        p=self.editor=self.page('模型配置')
        self.description=tk.StringVar()
        ttk.Label(p,textvariable=self.description,wraplength=950).grid(row=0,column=0,columnspan=3,sticky='w')
        frame=ttk.Frame(p);frame.grid(row=1,column=0,columnspan=3,sticky='nsew');p.rowconfigure(1,weight=1)
        self.tree=ttk.Treeview(frame,columns=('value','meaning'),height=8)
        self.tree.heading('#0',text='配置项');self.tree.heading('value',text='当前值（双击修改）');self.tree.heading('meaning',text='含义')
        self.tree.column('#0',width=210);self.tree.column('value',width=320);self.tree.column('meaning',width=360)
        self.tree.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(frame,command=self.tree.yview);scrollbar.pack(side='right',fill='y');self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind('<Double-1>',self.edit)
        buttons=ttk.Frame(p);buttons.grid(row=2,column=0,columnspan=3,sticky='w',pady=6)
        ttk.Button(buttons,text='从训练页刷新',command=self.refresh).pack(side='left')
        ttk.Button(buttons,text='应用到训练页',command=self.apply).pack(side='left',padx=8)
        ttk.Button(buttons,text='另存配置JSON',command=self.save).pack(side='left')
        self.render()

    def render(self):
        self.tree.delete(*self.tree.get_children());self.paths={}
        def insert(parent,obj,path=()):
            for key,value in obj.items():
                node=self.tree.insert(parent,'end',text=key,values=('' if isinstance(value,dict) else json.dumps(value,ensure_ascii=False),DESCRIPTIONS.get(key,'')),open=key in ('model','loss','training'))
                self.paths[node]=path+(key,)
                if isinstance(value,dict):insert(node,value,path+(key,))
        insert('',self.edit_config)
        m=self.edit_config['model'];d=self.edit_config['data']
        head='候选评分' if m['kind']=='scorer' else '直接分类头'
        self.description.set(f"结构预览：时延CNN → {m['layers']}层Transformer（宽度{m['d_model']}，{m['heads']}头，FFN={m['ffn_dim']}）→ SNR条件化 → {head}。时间池化={d['temporal_pool']}。改变结构后必须重新训练。")

    def refresh(self):
        self.edit_config=copy.deepcopy(self.app.config)
        try:
            self.edit_config['training'].update(epochs=int(self.app.epochs.get()),batch_size=int(self.app.batch.get()))
            self.edit_config['model']['layers']=int(self.app.layers.get())
            self.edit_config['data']['manifest']=self.app.manifest.get();self.edit_config['data']['candidates']=self.app.candidates.get()
            self.edit_config['data']['crop'].update(main_path_matlab=int(self.app.train_center.get()),already_cropped=self.app.train_cropped.get(),allow_short=self.app.train_short.get())
            self.edit_config['data']['cache_features']=self.app.cache.get()
        except ValueError:pass
        self.render()

    def edit(self,event):
        node=self.tree.identify_row(event.y)
        if not node:return
        path=self.paths[node];parent=self.edit_config
        for key in path[:-1]:parent=parent[key]
        old=parent[path[-1]]
        if isinstance(old,dict):return
        value=simpledialog.askstring('修改配置','.'.join(path)+'\n数值直接填写；布尔值填true/false；列表用JSON。',initialvalue=old if isinstance(old,str) else json.dumps(old),parent=self.app.root)
        if value is None:return
        try:
            new=value if isinstance(old,str) else json.loads(value)
            if isinstance(old,bool) and not isinstance(new,bool):raise ValueError('请填写true或false。')
            if isinstance(old,int) and not isinstance(old,bool) and (not isinstance(new,int) or isinstance(new,bool)):raise ValueError('此项需要整数。')
            parent[path[-1]]=new
            self.render()
        except Exception as error:messagebox.showerror('输入无效',str(error))

    def apply(self):
        try:
            validate_model_config(self.edit_config)
            path=ROOT/'configs'/('ui_model_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            write_json(path,self.edit_config);self.app.config_path.set(str(path));self.app.assign_config(self.edit_config)
            self.app.train_output.set(str(project_path(self.edit_config['training']['output'])))
            self.app.tabs.select(self.app.pages[1]);self.app.status.set('模型配置已保存并应用到训练页。')
        except Exception as error:messagebox.showerror('配置需要修正',str(error))

    def save(self):
        try:validate_model_config(self.edit_config)
        except Exception as error:messagebox.showerror('配置需要修正',str(error));return
        path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:write_json(path,self.edit_config)

    def add_simulation(self,mode,title):
        p=self.page(title);cfg=self.mcfg
        v={'page':p}
        for key,value in [('exe',cfg['matlab_exe']),('script',str(project_path(cfg[mode+'_script']))),
                          ('dependency',str(project_path(cfg['dependency_dir']))),('input',''),
                          ('output',str(ROOT/'outputs'/('matlab_'+mode)/'results.mat'))]:v[key]=tk.StringVar(value=value)
        self.app.entry(p,0,'MATLAB程序',v['exe'],'file');self.app.entry(p,1,'原始仿真脚本',v['script'],'file')
        self.app.entry(p,2,'依赖函数目录',v['dependency'],'dir');self.app.entry(p,3,'预测清单MAT',v['input'],'file')
        self.app.entry(p,4,'结果MAT完整路径',v['output'])
        ttk.Button(p,text='保存为…',command=lambda:self.pick_save(v['output'])).grid(row=4,column=2)
        settings=ttk.Frame(p);settings.grid(row=5,column=0,columnspan=3,sticky='w')
        for key,label,value in [('mc','MC次数',cfg['mc']),('center','主径',cfg['center']),('seed','种子',cfg['seed']),('limit','只跑前N信道（0=全部）',0)]:
            v[key]=tk.StringVar(value=str(value));ttk.Label(settings,text=label).pack(side='left');ttk.Entry(settings,textvariable=v[key],width=7).pack(side='left',padx=8)
        v['parallel']=tk.BooleanVar(value=cfg['parallel']);ttk.Checkbutton(settings,text='并行',variable=v['parallel']).pack(side='left')
        options=ttk.LabelFrame(p,text='保存内容（分析需要保留译码前MC、元信息和参数）',padding=6)
        options.grid(row=6,column=0,columnspan=3,sticky='ew',pady=6)
        v['save']={}
        entries=[('mc_coded','译码前MC BER'),('mc_decoded','译码后MC BER'),('mse','MSE'),('channel_H','信道H'),
                 ('symbols','发送/编码序列'),('tables','完整网格表'),('summary','汇总表'),('csv','另存表格CSV'),('xlsx','另存表格XLSX'),('all_workspace','全部工作区变量（可能很大）')]
        for i,(key,label) in enumerate(entries):
            v['save'][key]=tk.BooleanVar(value=cfg['save_options'].get(key,False))
            ttk.Checkbutton(options,text=label,variable=v['save'][key]).grid(row=i//5,column=i%5,sticky='w',padx=7)
        v['extra']=tk.StringVar();v['format']=tk.StringVar(value='-v7.3')
        ttk.Label(options,text='额外变量名（逗号分隔）').grid(row=2,column=0,sticky='w')
        ttk.Entry(options,textvariable=v['extra'],width=32).grid(row=2,column=1,columnspan=2,sticky='ew')
        ttk.Combobox(options,textvariable=v['format'],values=['-v7.3','-v7'],width=8,state='readonly').grid(row=2,column=3)
        ttk.Label(p,text='输入仍为信道路径/SNR清单；294页遍历全网格。额外变量只保存当时工作区的值，未生成的变量会记录为缺失。',wraplength=980).grid(row=7,column=0,columnspan=3,sticky='w')
        self.sim[mode]=v

    @staticmethod
    def pick_save(var):
        path=filedialog.asksaveasfilename(defaultextension='.mat',filetypes=[('MAT','*.mat')])
        if path:var.set(path)

    def matlab_job(self,mode):
        v=self.sim[mode];cfg=copy.deepcopy(self.mcfg)
        cfg.update(mode=mode,matlab_exe=v['exe'].get(),script=v['script'].get(),dependency_dir=v['dependency'].get(),
                   input_mat=v['input'].get(),output_mat=v['output'].get(),mc=int(v['mc'].get()),center=int(v['center'].get()),
                   seed=int(v['seed'].get()),max_channels=int(v['limit'].get()),parallel=v['parallel'].get())
        if not cfg['input_mat'] or not cfg['output_mat']:raise ValueError('请选择输入MAT与输出MAT。')
        if cfg['mc']<1 or cfg['center']<1 or cfg['max_channels']<0:raise ValueError('MC和主径需为正数；限制信道数不能负。')
        cfg['save_options']={k:value.get() for k,value in v['save'].items()}
        extra=[x.strip() for x in v['extra'].get().split(',') if x.strip()]
        cfg['save_options'].update(extra_variables=extra,mat_version=v['format'].get())
        return {'action':'matlab_simulation','matlab':cfg}

    def add_analysis(self):
        p=self.analysis=self.page('仿真结果分析')
        self.analysis_prediction=tk.StringVar();self.analysis_sim=tk.StringVar();self.analysis_grid=tk.StringVar()
        self.analysis_output=tk.StringVar(value=str(ROOT/'outputs/simulation_analysis'))
        self.fixed=tk.StringVar()
        for i,(label,var) in enumerate([('原预测清单MAT',self.analysis_prediction),('预测参数仿真MAT',self.analysis_sim),('完整网格仿真MAT',self.analysis_grid)]):self.app.entry(p,i,label,var,'file')
        self.app.entry(p,3,'分析结果根目录',self.analysis_output,'dir')
        self.app.entry(p,4,'固定参数 mu,N1,N2（可空）',self.fixed)
        ttk.Label(p,text='留空时取顶部模型记录的训练集固定参数。分析按信道名/SNR匹配，主要比较从网格内部抽取推荐候选的共同噪声结果。',wraplength=930).grid(row=5,column=0,columnspan=3,sticky='w',pady=8)
        ttk.Label(p,text='输出逐信道对比、异常排序、网格固定参数排名、PNG/PDF/SVG和HTML报告。关闭MC保存选项后可能无法分析。',wraplength=930).grid(row=6,column=0,columnspan=3,sticky='w')

    def analysis_job(self):
        for v in [self.analysis_prediction,self.analysis_sim,self.analysis_grid]:
            if not Path(v.get()).is_file():raise ValueError('请选择三份存在的MAT文件。')
        fixed=None
        if self.fixed.get().strip():
            fixed=[float(x.strip()) for x in self.fixed.get().replace('，',',').split(',')]
            if len(fixed)!=3 or not all(math.isfinite(x) for x in fixed):raise ValueError('固定参数应为mu,N1,N2三个数字。')
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return {'action':'simulation_analysis','prediction_mat':self.analysis_prediction.get(),'simulation_mat':self.analysis_sim.get(),
                'grid_mat':self.analysis_grid.get(),'checkpoint':self.app.checkpoint.get(),'fixed_parameters':fixed,
                'output':str(self.app.new_path(self.analysis_output.get(),stamp))}


def install(app):
    app.extensions=Extensions(app)

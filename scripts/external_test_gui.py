"""可重复使用的外部模型测试窗口；后台子进程执行，界面不会阻塞。"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from nlms_dfe.common import ROOT, read_config, write_json


class TestWindow:
    def __init__(self, root):
        self.root=root;self.process=None;self.events=queue.Queue();self.values={}
        root.title('外部模型测试工作台 — MeanCodedBER');root.geometry('1050x920')
        root.columnconfigure(1,weight=1)
        config=read_config('configs/external_test.json')
        for row,(key,label) in enumerate([('checkpoint','模型PT文件'),('channels','信道MAT目录'),('results','294组grid结果目录'),('output','测试输出根目录')]):
            value=tk.StringVar(value=config[key]);self.values[key]=value
            ttk.Label(root,text=label).grid(row=row,column=0,padx=8,pady=6)
            ttk.Entry(root,textvariable=value).grid(row=row,column=1,sticky='ew')
            ttk.Button(root,text='选择…',command=lambda k=key:self.browse(k)).grid(row=row,column=2,padx=8)
        self.center=tk.StringVar(value=str(config.get('main_path',626)))
        ttk.Label(root,text='原始主径列号').grid(row=4,column=0)
        ttk.Entry(root,textvariable=self.center).grid(row=4,column=1,sticky='w')
        self.short=tk.BooleanVar(value=config.get('allow_short',False));self.cropped=tk.BooleanVar(value=config.get('already_cropped',False))
        ttk.Checkbutton(root,text='不足4000行时取到实际末尾（记录差异）',variable=self.short).grid(row=5,column=1,sticky='w')
        ttk.Checkbutton(root,text='信道已按模型窗口裁剪',variable=self.cropped).grid(row=6,column=1,sticky='w')
        self.custom=tk.BooleanVar(value=config.get('fixed_parameters') is not None)
        ttk.Checkbutton(root,text='自定义固定参数（不勾选：新测试用模型基线；重画用结果中原基线）',variable=self.custom).grid(row=7,column=0,columnspan=3,sticky='w',padx=8)
        fields=ttk.Frame(root);fields.grid(row=8,column=0,columnspan=3,sticky='w',padx=8)
        self.fixed=[]
        for key,value in zip(('mu','N1','N2'),config.get('fixed_parameters') or [.2,20,20]):
            ttk.Label(fields,text=key).pack(side='left',padx=4)
            variable=tk.StringVar(value=str(value));self.fixed.append(variable)
            ttk.Entry(fields,textvariable=variable,width=10).pack(side='left')
        ttk.Label(root,text='使用真实SNR；固定组合必须在294组表中。绘图可更换基线，无需重新预测。').grid(row=9,column=0,columnspan=3,pady=8)
        self.plot_folder=tk.StringVar()
        ttk.Label(root,text='已完成测试结果目录').grid(row=10,column=0)
        ttk.Entry(root,textvariable=self.plot_folder).grid(row=10,column=1,sticky='ew')
        ttk.Button(root,text='选择…',command=self.choose_plot).grid(row=10,column=2)
        buttons=ttk.Frame(root);buttons.grid(row=11,column=0,columnspan=3)
        self.start=ttk.Button(buttons,text='开始测试',command=self.run);self.start.pack(side='left',padx=6)
        ttk.Button(buttons,text='停止本次测试',command=self.stop).pack(side='left',padx=6)
        ttk.Button(buttons,text='打开结果目录',command=lambda:os.startfile(self.values['output'].get())).pack(side='left',padx=6)
        ttk.Button(buttons,text='操作说明',command=lambda:os.startfile(ROOT/'docs/外部模型测试工作台说明.md')).pack(side='left',padx=6)
        ttk.Button(buttons,text='绘制结果图',command=self.plot).pack(side='left',padx=6)
        ttk.Button(buttons,text='加载指标表',command=self.load_table).pack(side='left',padx=6)
        self.summary_title=tk.StringVar(value='指标—结果：完成测试或选择已有结果后加载')
        ttk.Label(root,textvariable=self.summary_title,wraplength=1000).grid(row=12,column=0,columnspan=3,sticky='w',padx=8)
        table_frame=ttk.Frame(root);table_frame.grid(row=13,column=0,columnspan=3,sticky='nsew',padx=8,pady=4)
        self.summary=ttk.Treeview(table_frame,columns=('metric','value'),show='headings',height=13)
        self.summary.heading('metric',text='指标');self.summary.heading('value',text='结果')
        self.summary.column('metric',width=340);self.summary.column('value',width=580)
        self.summary.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(table_frame,orient='vertical',command=self.summary.yview);scrollbar.pack(side='right',fill='y');self.summary.configure(yscrollcommand=scrollbar.set)
        self.log=ScrolledText(root,height=7);self.log.grid(row=14,column=0,columnspan=3,sticky='nsew',padx=8,pady=8);root.rowconfigure(14,weight=1)
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(150,self.poll)

    def browse(self,key):
        path=filedialog.askopenfilename(filetypes=[('模型','*.pt')]) if key=='checkpoint' else filedialog.askdirectory()
        if path:self.values[key].set(path)

    def run(self):
        if self.process and self.process.poll() is None:return
        try:
            job={k:v.get() for k,v in self.values.items()}
            job.update(main_path=int(self.center.get()),allow_short=self.short.get(),already_cropped=self.cropped.get(),device='auto')
            job['fixed_parameters']=self.requested_fixed()
            if job['main_path']<1:raise ValueError('主径须为正整数')
            write_json(ROOT/'configs/external_test.json',job)
            self.launch(job)
        except Exception as error:messagebox.showerror('无法启动',str(error))

    def requested_fixed(self):
        if not self.custom.get():return None
        import math
        values=[float(v.get()) for v in self.fixed]
        if not all(math.isfinite(v) for v in values) or values[0]<=0 or values[1]<1 or values[2]<0 or any(v!=int(v) for v in values[1:]):
            raise ValueError('mu为正数，N1为正整数，N2为非负整数。')
        return values

    def choose_plot(self):
        path=filedialog.askdirectory(initialdir=self.values['output'].get())
        if path:
            self.plot_folder.set(path)
            self.load_table()

    def load_table(self,folder=None):
        """按已保存结果显示，未执行的输入框改动不会冒充已完成比较。"""
        from pathlib import Path
        try:
            path=Path(folder or self.plot_folder.get())
            file=path/'comparison_metrics.json'
            if not file.exists():file=path/'metrics.json'
            m=json.loads(file.read_text(encoding='utf-8-sig'))
            if 'source' in m:
                original=json.loads((Path(m['source'])/'metrics.json').read_text(encoding='utf-8-sig'))
                m={**original,**m}
            count=int(m['samples'])
            p=m['fixed_parameters']
            entries=[('信道数量',str(count)),('固定参数组合',f'mu={p[0]:g}，N1={p[1]:g}，N2={p[2]:g}')]
            for key,label in [('mean_selected_ber','模型平均 MeanCodedBER'),('mean_fixed_ber','固定参数平均 MeanCodedBER'),('mean_grid_best_ber','294组网格最低平均 BER')]:
                entries.append((label,f'{m[key]*100:.4f}%'))
            gain=m.get('relative_ber_reduction')
            entries.append(('相对固定参数的BER降低比例','不可计算（固定BER为0）' if gain is None else f'{gain*100:.2f}%'+('（负值表示变差）' if gain<0 else '')))
            for key,label in [('better_fraction','优于固定参数'),('worse_fraction','差于固定参数'),('tie_fraction','与固定参数持平')]:
                entries.append((label,f'{round(m[key]*count)}个信道（{m[key]*100:.2f}%）'))
            for key,label in [('top1','Top-1命中率'),('top3','Top-3命中率'),('top5','Top-5命中率')]:entries.append((label,f'{m[key]*100:.2f}%'))
            for key,label in [('mean_regret','平均 regret'),('p90_regret','P90 regret')]:entries.append((label,f'{m[key]:.6f}（{m[key]*100:.4f}个百分点）'))
            self.summary.delete(*self.summary.get_children())
            for row in entries:self.summary.insert('','end',values=row)
            self.summary_title.set(f'指标—结果（已保存数据）：{path}')
        except Exception as error:
            messagebox.showerror('无法加载指标',f'请选择完整结果目录。\n{error}')

    def plot(self):
        if self.process and self.process.poll() is None:return
        try:
            from pathlib import Path
            if not (Path(self.plot_folder.get())/'metrics.json').is_file():
                raise ValueError('请选择含metrics.json的单次已完成测试目录，不能选测试根目录。')
            self.launch(dict(action='plot',folder=self.plot_folder.get(),fixed_parameters=self.requested_fixed()))
        except Exception as error:messagebox.showerror('无法绘图',str(error))

    def launch(self,job):
        try:
            folder=ROOT/'outputs/external_test_jobs';folder.mkdir(parents=True,exist_ok=True)
            path=folder/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            write_json(path,job)
            self.process=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','scripts.external_model_test',str(path)],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            self.start.configure(state='disabled')
            def worker():
                for line in self.process.stdout:self.events.put(line)
                self.events.put(f'进程结束，退出码：{self.process.wait()}\n')
            threading.Thread(target=worker,daemon=True).start()
        except Exception as error:messagebox.showerror('无法启动',str(error))

    def poll(self):
        while not self.events.empty():
            line=self.events.get();self.log.insert('end',line);self.log.see('end')
            if line.startswith('完成：'):
                self.plot_folder.set(line.strip().split('：',1)[1]);self.load_table()
            if line.startswith('绘图完成：'):
                path=line.strip().split('：',1)[1];self.load_table(path);os.startfile(path)
        if self.process and self.process.poll() is not None:self.start.configure(state='normal')
        self.root.after(150,self.poll)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate();self.log.insert('end','已停止；已完成样本CSV保留，未完成结果不可当作全量结论。\n')

    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno('退出','停止当前测试并关闭窗口？'):return
            self.stop()
        self.root.destroy()


if __name__=='__main__':
    root=tk.Tk();TestWindow(root);root.mainloop()

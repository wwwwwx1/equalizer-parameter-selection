"""独立训练诊断入口；明确区分新版记录与旧版历史。"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox


def install(app):
    page=ttk.Frame(app.tabs,padding=12)
    page.columnconfigure(1,weight=1)
    # 放在训练页之后，确保在窗口较窄时也容易发现。
    app.tabs.add(page,text='训练诊断')
    # 保留原有页面索引，其他任务仍按原来的页号分发。
    folder=tk.StringVar()
    app.entry(page,0,'模型训练结果目录',folder,'dir')
    status=tk.StringVar(value='选择单个模型的结果目录；不是整个outputs目录。')
    ttk.Label(page,textvariable=status,wraplength=950).grid(row=2,column=0,columnspan=3,sticky='w',pady=10)
    ttk.Label(page,text='完整诊断：训练/验证BER、原始损失、加权贡献、Regret、命中率、学习率、梯度和耗时。\n首轮完成后生成诊断记录；旧记录只支持已有的epoch/batch曲线。',wraplength=950).grid(row=3,column=0,columnspan=3,sticky='w')
    def current():
        job=getattr(app,'active_job',{})
        if app.running and job.get('action')=='train':folder.set(job['config']['training']['output'])
        elif app.last_output:folder.set(str(app.last_output))
        else:status.set('尚无当前任务结果，请点击“选择…”指定结果目录。')
    def open_plot(kind):
        path=Path(folder.get())
        if not folder.get().strip() or not path.is_dir():
            messagebox.showinfo('请选择目录','请选择单个模型训练结果目录。');return
        try:
            if kind=='diagnostics':
                from scripts.diagnostic_plots import show
                show(app.root,path)
            elif kind=='history':
                from scripts.live_training_curves import show
                show(app.root,path)
            else:
                from scripts.diagnostic_plots import export
                export(path);status.set('已导出到 '+str(path/'figures'))
        except Exception as error:messagebox.showerror('诊断提示',str(error))
    buttons=ttk.Frame(page);buttons.grid(row=1,column=0,columnspan=3,sticky='w')
    for label,fn in [('使用本次结果',current),('打开完整实时诊断',lambda:open_plot('diagnostics')),('查看epoch / batch曲线',lambda:open_plot('history')),('导出诊断图',lambda:open_plot('export'))]:
        ttk.Button(buttons,text=label,command=fn).pack(side='left',padx=4)
    old_job=app.job
    def guide():
        from tkinter.scrolledtext import ScrolledText
        window=tk.Toplevel(app.root);window.title('诊断图逐图解读');window.geometry('900x700')
        text=ScrolledText(window,wrap='word',font=('Microsoft YaHei UI',11))
        text.pack(fill='both',expand=True)
        path=Path(__file__).resolve().parents[1]/'docs/诊断图逐图解读.md'
        text.insert('1.0',path.read_text(encoding='utf-8-sig'));text.configure(state='disabled')
    ttk.Button(page,text='每张图的含义与判断方法',command=guide).grid(row=4,column=0,columnspan=3,sticky='w',pady=12)
    def job():
        if app.tabs.select()==str(page):raise ValueError('诊断页面请使用页内的查看或导出按钮，无需开始训练任务。')
        return old_job()
    app.job=job
    app.diagnostics_page=page

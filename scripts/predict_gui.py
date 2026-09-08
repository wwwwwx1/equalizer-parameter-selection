"""给用户的简易预测窗口：选择MAT、填写SNR、显示并保存均衡参数。"""
import json
import math
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]


def predict_one(channel_path, snr, center=501, cropped=False, field='h_vary', allow_short=False):
    """后台执行；这里不访问窗口控件，避免界面卡住或线程错误。"""
    import torch
    from scipy.io import savemat
    from nlms_dfe.data.io import read_field
    from nlms_dfe.predict import Predictor
    from nlms_dfe.common import write_json, active_model, project_path

    torch.set_num_threads(4)
    active = active_model()
    checkpoint_path = project_path(active['checkpoint'])
    if not checkpoint_path.is_file():
        raise ValueError('尚未生成新模型，请先完成数据准备和训练。')
    model = Predictor(checkpoint_path)
    if model.checkpoint['config']['training'].get('target_metric','MeanDecodedBER') != active['target_metric']:
        raise ValueError('模型与配置中的BER目标不一致，请检查模型入口。')
    h = read_field(channel_path, field, model.checkpoint['config']['data']['hdf5_matlab_order'])
    result = model.predict(h, snr, already_cropped=cropped,
                           main_path_matlab=None if cropped else center, allow_short=allow_short)
    folder = ROOT/'outputs/预测结果'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    result.update(channel_path=str(channel_path), checkpoint=active['checkpoint'], snr_db=snr, main_path_matlab=None if cropped else center,
                  already_cropped=cropped, allow_short=allow_short, channel_field=field)
    write_json(folder/'推荐参数.json', result)
    selected = result['selected']
    savemat(folder/'推荐参数.mat', {k:selected[k] for k in ('mu','N1','N2','candidate_id')})
    return result, folder


class PredictionWindow:
    def __init__(self, root):
        self.root = root
        self.messages = queue.Queue()
        self.folder = None
        self.matlab_text = ''
        from nlms_dfe.common import active_model
        root.title('信道均衡器参数预测 — ' + active_model()['label'])
        root.geometry('800x630')
        root.minsize(740, 610)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 18, 'bold'))
        style.configure('Result.TLabel', font=('Microsoft YaHei UI', 19, 'bold'))
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text='输入信道和信噪比，得到推荐参数', style='Title.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 18))
        ttk.Label(frame, text='1. 选择信道文件（包含 h_vary 的原始 MAT）').grid(row=1, column=0, columnspan=2, sticky='w')
        self.path = tk.StringVar()
        ttk.Entry(frame, textvariable=self.path).grid(row=2, column=0, sticky='ew', pady=8)
        ttk.Button(frame, text='选择文件…', command=self.browse).grid(row=2, column=1, padx=(10, 0))
        inputs = ttk.Frame(frame)
        inputs.grid(row=3, column=0, columnspan=2, sticky='w', pady=(8, 16))
        ttk.Label(inputs, text='2. 信噪比（dB）：').pack(side='left')
        self.snr = tk.StringVar()
        ttk.Entry(inputs, textvariable=self.snr, width=12).pack(side='left')
        ttk.Label(inputs, text='   可填写负数，例如 -3.5').pack(side='left')
        settings = ttk.LabelFrame(frame, text='信道设置（通常保持默认）', padding=10)
        settings.grid(row=4, column=0, columnspan=2, sticky='ew')
        ttk.Label(settings, text='原始信道主径列号：').grid(row=0, column=0, sticky='w')
        self.center = tk.StringVar(value='501')
        ttk.Combobox(settings, textvariable=self.center, values=('501','626'), width=9, state='readonly').grid(row=0, column=1, sticky='w')
        ttk.Label(settings, text='新电脑默认501；若原始主径不同请调整。').grid(row=0, column=2, padx=10)
        self.cropped = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text='文件已经裁好（时间 × 501列），不再裁剪', variable=self.cropped).grid(row=1, column=0, columnspan=3, sticky='w', pady=4)
        self.short = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text='不足4000行时，允许使用实际末尾', variable=self.short).grid(row=2, column=0, columnspan=3, sticky='w')
        ttk.Label(settings, text='变量名：').grid(row=3, column=0, sticky='w', pady=4)
        self.field = tk.StringVar(value='h_vary')
        ttk.Entry(settings, textvariable=self.field, width=16).grid(row=3, column=1, sticky='w')
        ttk.Label(settings, text='默认窗口：626:4000，主径前后各250点').grid(row=3, column=2, sticky='w', padx=10)
        self.run_button = ttk.Button(frame, text='3. 开始预测', command=self.start)
        self.run_button.grid(row=5, column=0, columnspan=2, sticky='ew', pady=16)
        self.status = tk.StringVar(value='已准备好。请选择文件并填写该信道的SNR。')
        ttk.Label(frame, textvariable=self.status, wraplength=720).grid(row=6, column=0, columnspan=2, sticky='w')
        self.answer = tk.StringVar(value='μ = —     N1 = —     N2 = —')
        ttk.Label(frame, textvariable=self.answer, style='Result.TLabel').grid(row=7, column=0, columnspan=2, sticky='w', pady=18)
        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, columnspan=2, sticky='w')
        self.copy_button = ttk.Button(actions, text='复制MATLAB参数', command=self.copy, state='disabled')
        self.copy_button.pack(side='left')
        self.folder_button = ttk.Button(actions, text='打开结果文件夹', command=self.open_folder, state='disabled')
        self.folder_button.pack(side='left', padx=12)
        ttk.Button(actions, text='查看训练图', command=self.open_training_figures).pack(side='left')
        ttk.Label(frame, text='得到参数后，在您的仿真中与固定参数比较。此窗口不会计算新信道的BER。', wraplength=710).grid(row=9, column=0, columnspan=2, sticky='w', pady=(16, 0))
        root.after(100, self.poll)

    def browse(self):
        path = filedialog.askopenfilename(title='请选择原始信道MAT文件', initialdir=str(ROOT),
                                          filetypes=[('信道文件', '*.mat *.npz *.h5'), ('所有文件','*.*')])
        if path:
            self.path.set(path)

    def start(self):
        try:
            path = Path(self.path.get().strip())
            if not path.is_file():
                raise ValueError('请先选择一个存在的信道文件。')
            snr = float(self.snr.get().strip())
            if not math.isfinite(snr):
                raise ValueError('信噪比需要是有限数字。')
            args = (path, snr, int(self.center.get()), self.cropped.get(), self.field.get().strip(), self.short.get())
        except ValueError:
            messagebox.showerror('请检查输入', '请选择有效信道文件，并填写数字形式的信噪比，例如 -3.5 或 5。')
            return
        self.run_button.configure(state='disabled')
        self.copy_button.configure(state='disabled')
        self.folder_button.configure(state='disabled')
        self.answer.set('正在计算…')
        self.status.set('正在读取信道并预测，请稍候。首次启动需要加载模型。')
        threading.Thread(target=self.worker, args=args, daemon=True).start()

    def worker(self, *args):
        try:
            self.messages.put(('ok', predict_one(*args)))
        except Exception as error:
            self.messages.put(('error', str(error)))

    def poll(self):
        try:
            state, payload = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll)
            return
        self.run_button.configure(state='normal')
        if state == 'ok':
            result, self.folder = payload
            p = result['selected']
            mu = format(p['mu'], '.7g')
            self.answer.set(f"μ = {mu}     N1 = {p['N1']}     N2 = {p['N2']}")
            self.matlab_text = f"mu = {mu};\nN1 = {p['N1']};\nN2 = {p['N2']};"
            self.status.set('预测完成，已自动保存“推荐参数.mat”和“推荐参数.json”。')
            self.copy_button.configure(state='normal')
            self.folder_button.configure(state='normal')
        else:
            self.answer.set('未得到结果')
            self.status.set('请检查文件和信道设置后重试。')
            messagebox.showerror('预测未完成', payload + '\n\n请确认选择的是信道文件，不是只含BER的结果文件。')
        self.root.after(100, self.poll)

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.matlab_text)
        self.status.set('已复制。可直接粘贴到MATLAB脚本或命令窗口。')

    def open_folder(self):
        if self.folder:
            os.startfile(self.folder)

    def open_training_figures(self):
        from nlms_dfe.common import active_model, project_path
        folder = project_path(active_model()['checkpoint']).parent/'figures'
        if folder.exists():
            os.startfile(folder)
        else:
            messagebox.showinfo('训练图', '本轮训练图尚未生成，训练完成后即可查看。')


def main():
    root = tk.Tk()
    PredictionWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()

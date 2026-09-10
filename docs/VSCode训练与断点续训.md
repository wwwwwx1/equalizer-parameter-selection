# VS Code训练、停止与恢复

## 环境与路径

在VS Code用“文件→打开文件夹”打开实际项目根目录。按Ctrl+Shift+P，选择Python: Select Interpreter，选择已经安装PyTorch的python.exe。新电脑离线环境不需要重复联网安装，但需已有torch、numpy、scipy、h5py、matplotlib、tkinter。

以下命令在VS Code的终端执行（终端当前目录必须是项目根目录）。先检查解释器：

```powershell
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

如果python指向错误环境，PowerShell可以显式调用，例如本机：

```powershell
& 'C:\Compilation\_environment\python\shared-pytorch-cuda\Scripts\python.exe' -m scripts.train_cli --config configs/train_window.json --epochs 10 --batch-size 8
```

新电脑将解释器路径替换为自己的python.exe。

## 新训练

先在工作台准备数据，得到数据目录中的train_config.json；确认信道主径（原始主径501或626必须按数据填写）、已裁剪标志及MeanCodedBER映射。

```powershell
python -m scripts.train_cli --config "data/window_prepared/你的数据目录/train_config.json" --epochs 10 --batch-size 8
```

新的CLI默认按来源组7:2:1生成独立索引，不修改原始索引。自动创建outputs/cli_training/时间戳结果目录。轮数和批次命令行参数优先于配置。默认最终测试不用于训练选模，命令行不会修改全局预测模型入口。

## 暂停、停止和续训

工作台新任务支持“暂停”“继续”“保存并停止”；暂停在读取完成后的批次边界生效，可能要等当前批次结束。多模型模式按钮控制整套任务；停止时尚未启动的任务也会在检查点退出。

终端按Ctrl+C可以停止。不要用强制结束进程或关机替代。last.pt在初始化完成后及每个完整epoch结束原子更新；best.pt保留验证选出的权重。全量模式也保存last.pt，不再只能等最后一轮。

停止时保留最近完整轮次，当前未完成的一轮重跑，不支持逐batch精确续训。初始化扫描尚未完成就停止时没有可恢复权重。停止后训练不会自动执行最终测试。

```powershell
python -m scripts.train_cli --resume "outputs/cli_training/上次目录/last.pt"
```

恢复创建新的输出目录，读取断点原始配置，保留原目录；恢复模型、优化器、学习率调度、AMP缩放、随机状态、数据加载器顺序及早停状态。不要在恢复命令中改epochs、batch-size；本版要求原训练计划保持一致。旧版best.pt/model.pt没有这些完整状态，不能直接续训。续训完成后按原验证选模规则评估测试集。

工作台普通训练页也可以选择“断点last.pt”；此时按断点的原设置恢复，界面中的新轮数等不会覆盖它，启动日志显示实际参数。

命令行需要暂停功能时，启动加 --control-file outputs/control.json，在另一个PowerShell终端执行以下之一：

```powershell
Set-Content -Encoding utf8 outputs/control.json '{"action":"pause"}'
Set-Content -Encoding utf8 outputs/control.json '{"action":"run"}'
Set-Content -Encoding utf8 outputs/control.json '{"action":"stop"}'
```

再次启动任务前将控制文件设回run。工作台会为每套任务新建控制文件。

## 查看中间曲线

工作台点击“中途曲线”，选择实际模型训练目录。也可另开终端：

```powershell
python -m scripts.live_training_curves --output "outputs/cli_training/本次目录"
```

每两秒刷新；显示已完成epoch的Loss、验证BER和固定基线、Regret，以及最近3000个batch的原始Loss和20批滑动平均。完整batch记录保存在batch_history.csv，包含加权回归、软标签、期望Regret、排序、分类损失分量。training_history.csv保留完整轮次记录。

预热前后损失定义不同，不能用总Loss跳变判断发散；评估泛化应看验证BER及最终独立测试。中途曲线可在训练没有完成或已经停止时查看；续训输出的batch文件只记录续训阶段，旧批次记录留在原目录。

## 本次更新边界

训练机制更新不改变既有模型权重，也不保证BER自动改善。默认普通训练和多模型组外模式均为7:2:1来源组划分；全量训练仍可显式选择。当前新增模块沿用原Python依赖。已有正在运行的旧进程不会自动获得新代码功能，不要为了更新强制中断它。

## 怎样判断训练是否有效

总Loss只用于检查数值稳定性；它不是最终目标。每轮优先看验证集MeanCodedBER是否低于固定参数BER、验证Regret是否下降，以及训练Loss下降时验证BER是否反而持续升高。后一种模式才是过拟合的直接信号。测试集只在结构和轮数确定后读取一次。

新配置默认在3轮回归预热后，用5轮把软标签、Regret和排序项逐步加到完整权重。这样不会再在第4轮把附加损失全部突变加入。学习率默认使用plateau策略：验证BER连续3轮没有改善时减半，最低到1e-6；不是按预定80轮机械地持续衰减。模型配置页可修改loss_ramp_epochs、lr_schedule、lr_factor、lr_patience和min_learning_rate。调试时建议先设10轮、batch_size=8或16（显存允许时），再根据验证BER决定是否扩大训练。

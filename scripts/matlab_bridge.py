"""将用户MATLAB脚本的运行副本参数化，通过matlab -batch执行。"""
import hashlib
import json
import re
import subprocess
import os
from datetime import datetime
from pathlib import Path
from nlms_dfe.common import ROOT,project_path,write_json


def replace_once(text,pattern,replacement):
    text,count=re.subn(pattern,lambda m:replacement,text,count=1,flags=re.MULTILINE)
    if count!=1:raise ValueError('MATLAB脚本结构与适配器不符，未找到：'+pattern)
    return text


def prepare_script(config,folder):
    original=project_path(config['script'])
    text=original.read_text(encoding='utf-8-sig')
    config['source_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
    text=re.sub(r'^\s*(clear all|close all|clc);\s*$', '',text,flags=re.MULTILINE)
    text=replace_once(text,r'^\s*rng\(22\);','rng(WBconfig.seed);')
    text=replace_once(text,r"^predictionMatFile\s*=.*?;",'predictionMatFile = WBconfig.input_mat;')
    text=replace_once(text,r"^saveDir\s*=.*?;",'saveDir = fileparts(WBconfig.output_mat);')
    text=replace_once(text,r"^allResultMatFile\s*=.*?;",'allResultMatFile = WBconfig.output_mat;')
    text=replace_once(text,r"^addpath\(.*?\);",'addpath(WBconfig.dependency_dir);')
    text=replace_once(text,r'^MC\s*=\s*25;', 'MC = WBconfig.mc;')
    text=replace_once(text,r'^\s*Index0_center\s*=\s*626;', '        Index0_center = WBconfig.center;')
    if not config['parallel']:
        text=re.sub(r'^(\s*)parfor\s+',r'\1for ',text,flags=re.MULTILINE)
    # 仅限前N个信道的试运行；0表示全部。原输入文件不修改。
    text=replace_once(text,r'^numFiles\s*=\s*numel\(channel_paths\);',
                      'numFiles = numel(channel_paths);\nif WBconfig.max_channels > 0, numFiles = min(numFiles, WBconfig.max_channels); end')
    text=replace_once(text,r'^for kk = 1:numFiles', 'for kk = 1:numFiles\n    wb_progress(kk-1,numFiles);')
    text,saves=re.subn(r'(?m)^(\s*)save\(allResultMatFile,',r'\1wb_save(allResultMatFile,',text)
    if not saves:raise ValueError('未找到原脚本保存调用，不能保证保存选项生效。')
    key='simulation_results' if config['mode']=='prediction' else 'all_grid_results'
    location=re.search(r'^function\s',text,flags=re.MULTILINE)
    if not location:raise ValueError('没有找到原脚本局部函数边界。')
    text=text[:location.start()]+f'wb_finish(WBconfig,{key});\n\n'+text[location.start():]
    cfg_path=folder/'matlab_config.json'
    write_json(cfg_path,config)
    quote=lambda x:str(x).replace("'","''").replace('\\','/')
    text="WBconfig = jsondecode(fileread('"+quote(cfg_path)+"'));\n"+text
    generated=folder/'workbench_simulation.m'
    generated.write_text(text,encoding='utf-8')
    return generated


def run_matlab(job):
    config=dict(job['matlab'])
    config['save_options'].setdefault('xlsx',False)
    output=project_path(config['output_mat'])
    if output.exists():raise ValueError('结果MAT已存在，请选择新文件名，避免覆盖。')
    if not project_path(config['input_mat']).is_file():raise ValueError('预测输入MAT不存在。')
    exe=Path(config['matlab_exe'])
    if not exe.is_file():raise ValueError('请选择新电脑实际的matlab.exe。')
    for key in ['input_mat','output_mat','dependency_dir','script']:
        config[key]=str(project_path(config[key]))
    if not Path(config['dependency_dir']).is_dir():raise ValueError('MATLAB依赖目录不存在。')
    config['dependency_sha256']={}
    for name in ['shift_H_pd','shift_H_pd_all']:
        dependency=Path(config['dependency_dir'])/(name+'.m')
        if not dependency.is_file():raise ValueError('依赖目录缺少：'+dependency.name)
        config['dependency_sha256'][name]=hashlib.sha256(dependency.read_bytes()).hexdigest()
    if config['mc']<1 or config['center']<1 or config['max_channels']<0:raise ValueError('MC/主径/试运行信道数无效。')
    for name in config['save_options']['extra_variables']:
        if not re.fullmatch(r'[A-Za-z]\w*',name):raise ValueError('额外变量名必须为普通MATLAB变量标识符。')
    if config['save_options']['mat_version'] not in ('-v7','-v7.3'):raise ValueError('MAT版本只支持v7或v7.3。')
    output.parent.mkdir(parents=True,exist_ok=True)
    # 同一个输出文件名可能因上次失败而留下运行日志目录；
    # 每次运行使用独立记录目录，不把“日志目录已存在”误判为仿真错误。
    folder=output.parent/(output.stem+'_workbench_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(exist_ok=False)
    config['status_file']=str(folder/'status.json')
    generated=prepare_script(config,folder)
    quote=lambda s:str(s).replace('\\','/').replace("'","''")
    command=f"addpath('{quote(ROOT/'matlab_integration')}'); run('{quote(generated)}');"
    args=[str(exe),'-batch',command,'-logfile',str(folder/'matlab.log')]
    print('启动MATLAB：'+config['mode']+'，输出：'+str(output),flush=True)
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    for raw in process.stdout:
        try:line=raw.decode('utf-8')
        except UnicodeDecodeError:line=raw.decode('gb18030',errors='replace')
        line=line.replace('\r\n','\n').replace('\r','')
        print(line,end='',flush=True)
    code=process.wait()
    if code!=0:raise RuntimeError(f'MATLAB退出码{code}，详细日志：{folder/"matlab.log"}')
    status=Path(config['status_file'])
    if not status.is_file() or not output.is_file():raise RuntimeError('MATLAB未生成最终状态或MAT结果。')
    report=json.loads(status.read_text(encoding='utf-8-sig'))
    missing=report.get('missing_extra_variables',[])
    return {'action':'matlab_'+config['mode'],'output':str(output.parent),'output_mat':str(output),
            'summary':report,'message':f"MATLAB完成：成功{report['successful']}、失败{report['failed']}。结果已保存。"+(f'未生成的额外变量：{missing}' if missing else '')}

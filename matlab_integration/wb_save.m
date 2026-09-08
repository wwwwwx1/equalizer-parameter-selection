function wb_save(filename, varargin)
% 工作台保存适配器：筛选原脚本保存变量，不改变通信计算。
cfg = evalin('caller','WBconfig');
opt = cfg.save_options;
names = varargin(~startsWith(string(varargin), '-'));
if opt.all_workspace
    names = evalin('caller','who');
end
extra = cellstr(string(opt.extra_variables));
names = unique([cellstr(string(names(:))); extra(:)], 'stable');
S = struct();
for i = 1:numel(names)
    name = names{i};
    if isempty(name), continue; end
    assert(isvarname(name),'非法保存变量名：%s',name);
    if ~evalin('caller',sprintf('exist(''%s'',''var'')',name))
        % 网格中间保存时某些汇总变量尚未生成；保留清单供最终保存。
        continue;
    end
    if ~opt.symbols && any(strcmp(name,{'msg_source','code_data','ss'})), continue; end
    if ~opt.summary && strcmp(name,'summaryTable'), continue; end
    if ~opt.tables && strcmp(name,'all_param_results'), continue; end
    value = evalin('caller',name);
    if isstruct(value) && any(strcmp(name,{'simulation_results','all_grid_results'}))
        fields = fieldnames(value);
        remove = {};
        for j = 1:numel(fields)
            f = fields{j};
            if (~opt.mc_coded && startsWith(f,'coded_ber_') && endsWith(f,'_mc')) || ...
               (~opt.mc_decoded && startsWith(f,'decoded_ber_') && endsWith(f,'_mc')) || ...
               (~opt.mse && contains(lower(f),'mse')) || ...
               (~opt.channel_H && strcmp(f,'H')) || ...
               (~opt.tables && any(strcmp(f,{'resultTable','resultTableSorted','bestResult'})))
                remove{end+1} = f; %#ok<AGROW>
            end
        end
        if ~isempty(remove), value = rmfield(value,remove); end
    end
    S.(name) = value;
end
S.workbench_config = cfg;
missing = {};
for i = 1:numel(extra)
    if ~isempty(extra{i}) && isvarname(extra{i}) && ~evalin('caller',sprintf('exist(''%s'',''var'')',extra{i}))
        missing{end+1} = extra{i}; %#ok<AGROW>
    end
end
S.workbench_missing_extra_variables = missing;
save(filename,'-struct','S',opt.mat_version);
if opt.csv || (isfield(opt,'xlsx') && opt.xlsx)
    for tableName = {'summaryTable','all_param_results'}
        name = tableName{1};
        if evalin('caller',sprintf('exist(''%s'',''var'')',name))
            T = evalin('caller',name);
            if istable(T)
                [folder,base,~] = fileparts(filename);
                if opt.csv, writetable(T,fullfile(folder,[base '_' name '.csv'])); end
                if isfield(opt,'xlsx') && opt.xlsx, writetable(T,fullfile(folder,[base '_' name '.xlsx'])); end
            end
        end
    end
end
end

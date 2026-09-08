function wb_finish(cfg, results)
ok = [results.simulation_success];
report = struct('total',numel(results),'successful',sum(ok),'failed',sum(~ok),...
    'output_mat',cfg.output_mat);
extra = cellstr(string(cfg.save_options.extra_variables));
missing = {};
for i=1:numel(extra)
    if ~isempty(extra{i}) && ~evalin('caller',sprintf('exist(''%s'',''var'')',extra{i}))
        missing{end+1}=extra{i}; %#ok<AGROW>
    end
end
report.missing_extra_variables = missing;
fid=fopen(cfg.status_file,'w','n','UTF-8');
assert(fid~=-1,'不能保存工作台状态');
fprintf(fid,'%s',jsonencode(report));fclose(fid);
wb_progress(numel(results),numel(results));
fprintf('WORKBENCH COMPLETE: %d success, %d failed.\n',sum(ok),sum(~ok));
end

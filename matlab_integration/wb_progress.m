function wb_progress(current,total)
p = struct('kind','progress','stage','MATLAB simulation','current',current,...
    'total',total,'percent',100*current/max(total,1),'detail','Channel progress');
fprintf('@@NLMS@@%s\n',jsonencode(p));
end

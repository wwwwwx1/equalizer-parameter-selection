function H_pd_all = shift_H_pd_all(H_mat, fs, fc, fss)
% H_mat: numRow × L，每一行是一条信道；输出同结构

    L = size(H_mat,2);

    % 1) 约简重采样比（原代码里 N_channel 约掉了，比值就是 fs/fss）
    g = gcd(fs, fss);
    p = fs/g;   q = fss/g;              % 4000/125 → 32/1

    % 2) resample 对矩阵按"列"处理，所以转置：每行信道变成一列
    %    ★ 滤波器只设计这一次，然后一次性处理所有信道
    re_all = resample(H_mat.', p, q);   % L2 × numRow
    L2 = size(re_all,1);

    % 3) 上载波：复指数只算一次，广播相乘后取实
    c = exp(1i*2*pi*fc*(1:L2).'/fs);    % L2×1，对应原来的 ts=(1:length)/fs
    H_pd_all = real(re_all .* c).';     % R2016b+ 隐式广播
    % 旧版本用:  H_pd_all = real(bsxfun(@times, re_all, c)).';
end

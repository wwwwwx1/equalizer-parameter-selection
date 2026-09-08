function H_pd = shift_H_pd(H,fs,fc)
    N_channel = length(H);
    % x=1:length(H)
    % figure
    % plot(x,H)
    re_H = resample(H, fs/125*N_channel, N_channel); %%采样率：从 125Hz → 恢复成原始高采样率 fs
    % y=1:length(re_H)
    % figure
    % plot(y,re_H)
    ts = (1:length(re_H))/fs;
    sig = re_H.*exp(1i*2*pi*fc*ts);% 上载波
    y1 = fft(sig);
    y = zeros(1,length(y1));
    % y(1:fc*2*length(y1)/fs)=(y1(1:fc*2*length(y1)/fs));% 去掉flip
    % y(length(y1)-fc*2*length(y1)/fs+1:length(y1))=(y1(length(y1)-fc*2*length(y1)/fs+1:length(y1)));% 去掉flip
    N = round( fc * 2 * length(y1) / fs ); 
    % 防护：防止索引超出数组范围
    N = max(1, min(N, length(y1))); 
    
    % ===================== 修复点 2 =====================
    % 用整数N替代浮点数表达式，干净无警告
    y(1:N) = y1(1:N); % 截取前N个点
    y(end-N+1:end) = y1(end-N+1:end); % 截取后N个点（end是MATLAB语法糖，更简洁）
    
    
    H_h = ifft(y);
    h_fc =H_h;
    H_pd = real(h_fc);% 频带信道
end
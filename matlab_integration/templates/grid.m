clear all;
close all;
clc;
tic
rng(22);                                  % 固定随机种子，便于复现
% parpool('local');                        % 如需可手动开启并行池

%% ====================== 读取文件名、路径和SNR ======================
% 这里只从 batch_predictions.mat 读取3个变量：
% channel_names、channel_paths、snr_db
% 不读取 rate、ber，也不读取预测的 mu/N1/N2 等其它字段
predictionMatFile = 'D:\机器学习\均衡器参数选择\批量信道预测\结果\20260906_212914_134057\batch_predictions.mat';
inputData = load(predictionMatFile,'channel_names','channel_paths','snr_db');

channel_names = inputData.channel_names;
channel_paths = inputData.channel_paths;
snr_db        = inputData.snr_db;
numFiles = numel(channel_paths);

%% ====================== 最终只保存一个MAT文件 ======================
saveDir = 'D:\matlab_code\Batch_BPSK\grid_all_results\';
if ~exist(saveDir,'dir')
    mkdir(saveDir);
end
allResultMatFile = fullfile(saveDir,'all_channels_294_grid_results.mat');

addpath('D:\matlab_code\Batch_BPSK\to zrl');

%% ====================== 参数设定 ======================
M = 2;                              % 二相调制
k = log2(M);
Rb = 32;                            % 比特率
Rs = Rb/k;                          % 符号率
beta = 1;                           % 滚降因子
span = 6;                           % 滤波器跨度
fc = 450;                           % 载波频率
fs = 4000;
sps2 = round(fs/Rb);
fss = 125;
MC = 25;                            % 每组参数Monte Carlo次数

% 信道编码
L = 7;
tblen = 3*L;
trellis = poly2trellis(7,[133 171]);

% 发送数据长度
N = 200;
train_ratio = 0.4;

%% ====================== 294组网格参数 ======================
delt_set = [0.2 0.3 0.35 0.4 0.5 0.6];
N1_set = 10:10:70;
N2_set = 20:10:80;
numParam = numel(delt_set)*numel(N1_set)*numel(N2_set);   % 6*7*7 = 294

param_delt = zeros(numParam,1);
param_N1 = zeros(numParam,1);
param_N2 = zeros(numParam,1);
p = 0;
for id = 1:numel(delt_set)
    for i1 = 1:numel(N1_set)
        for i2 = 1:numel(N2_set)
            p = p + 1;
            param_delt(p) = delt_set(id);
            param_N1(p) = N1_set(i1);
            param_N2(p) = N2_set(i2);
        end
    end
end
fprintf('参数网格总数 = %d\n',numParam);

%% ====================== 固定发送序列 ======================
% 所有信道、所有MC、所有均衡器参数都使用同一套发送数据
msg_source = randi([0 M-1],1,N);
code_data = convenc(msg_source,trellis);
Nc = length(code_data);
aa = code_data;
ss = exp(1j*aa*pi);
star = [1+0*1i,-1+0*1i];

num = round(train_ratio*Nc);
num = 2*floor(num/2);
num_info = num/2;

%% ====================== 发射机 ======================
up_ss_ch = upsample(ss,sps2);
rcos_fir = rcosdesign(beta,span,sps2);
rcos_ss_ch = conv(up_ss_ch,rcos_fir,'same');
t = (1:length(rcos_ss_ch))/fs;
tx_signal = rcos_ss_ch.*exp(1j*2*pi*fc.*t);
tx_signal = real(tx_signal);
fir_lp = fir1(128,0.2);

%% ====================== 初始化所有信道结果 ======================
all_grid_results = repmat(struct( ...
    'channel_name','', ...
    'channel_path','', ...
    'snr_db',NaN, ...
    'simulation_success',false, ...
    'error_message','', ...
    'H',[], ...
    'coded_ber_before_mc',[], ...
    'decoded_ber_before_mc',[], ...
    'coded_ber_after_mc',[], ...
    'decoded_ber_after_mc',[], ...
    'mse_train_mc',[], ...
    'mse_decision_mc',[], ...
    'param_delt',[], ...
    'param_N1',[], ...
    'param_N2',[], ...
    'resultTable',table(), ...
    'resultTableSorted',table(), ...
    'bestResult',table()), numFiles,1);

% 所有信道的294组统计结果统一拼接到这个表中
% 若共有26个信道，成功时最终应有 26*294 = 7644 行
all_param_results = table();

%% ====================== 逐个信道进行294组网格仿真 ======================
for kk = 1:numFiles
    fprintf('\n============================================================\n');
    fprintf('处理信道 %d / %d\n',kk,numFiles);

    % ---------- 当前文件名、路径、SNR ----------
    if iscell(channel_names)
        currName = char(channel_names{kk});
    elseif isstring(channel_names)
        currName = char(channel_names(kk));
    else
        currName = char(channel_names(kk,:));
    end

    if iscell(channel_paths)
        filePath = char(channel_paths{kk});
    elseif isstring(channel_paths)
        filePath = char(channel_paths(kk));
    else
        filePath = char(channel_paths(kk,:));
    end

    currSNR = double(snr_db(kk));
    fprintf('信道文件：%s\n',currName);
    fprintf('信道路径：%s\n',filePath);
    fprintf('当前SNR = %.4f dB\n',currSNR);

    all_grid_results(kk).channel_name = currName;
    all_grid_results(kk).channel_path = filePath;
    all_grid_results(kk).snr_db = currSNR;

    try
        %% ---------- 读取当前信道 ----------
        channelData = load(filePath,'h_vary');
        if ~isfield(channelData,'h_vary')
            error('当前信道MAT文件中不存在 h_vary 变量。');
        end
        h_vary = channelData.h_vary;

        %% ---------- 原始信道处理 ----------
        Index0_center = 626;
        win_side = round(2*fss);
        startIdx = Index0_center - win_side;
        endIdx   = min(size(h_vary,2),Index0_center + win_side);
        h_vary = h_vary(:,startIdx:endIdx);

        H = h_vary(end,:);
        H_pd = shift_H_pd(H,fs,fc);
        Index0 = find(H_pd == max(H_pd));
        NNa = find(H_pd == max(H_pd)) - 1;
        NNc = size(H_pd,2) - Index0;
        h_vary_pd = shift_H_pd_all(h_vary(626:end,:),fs,fc,fss);

        %% ---------- 发射信号通过当前动态信道 ----------
        Data = tx_signal;
        Signal = 0 * Data;
        dt = 0.8/100;
        H_Filter = h_vary_pd;
        Data0 = [zeros(1,NNa),zeros(1,NNc),Data,zeros(1,NNa),zeros(1,NNc)];

        for ii = 1:length(Data)+length(H_pd)-1
            IndexBegin = floor((ii-1)/(dt*fs)) + 1;
            h1 = H_Filter(IndexBegin,:);
            h2 = H_Filter(IndexBegin+1,:);
            t_interp = ii - (IndexBegin-1)*dt*fs - 1;
            h = h1 + (h2-h1)./(dt*fs)*t_interp;
            Signal(ii) = Data0(ii:ii+NNa+NNc) * transpose(flip(h));
        end

        Posmaxh = find(abs(h)==max(abs(h)));
        Signal = Signal(Posmaxh:end-length(h)+Posmaxh);
        Signal = real(Signal);
        Signal = Signal./max(abs(Signal));
        rx_clean = Signal;

        %% ---------- 当前信道结果矩阵 ----------
        coded_ber_after_mc   = nan(numParam,MC);
        decoded_ber_after_mc = nan(numParam,MC);
        mse_train_mc         = nan(numParam,MC);
        mse_decision_mc      = nan(numParam,MC);
        coded_ber_before_mc   = nan(1,MC);
        decoded_ber_before_mc = nan(1,MC);

        %% ---------- 预生成全部MC接收信号 ----------
        % 同一次MC下，294组参数使用完全相同的噪声，保持公平比较
        xx_all = zeros(MC,Nc);
        for mc = 1:MC
            ch_p = awgn(rx_clean,currSNR,'measured');
            xx1 = ch_p.*exp(-1j*2*pi*fc.*t);
            rcos_ss_1p = conv(xx1,fir_lp,'same');
            xx2 = conv(rcos_ss_1p,rcos_fir,'same');
            xx3 = xx2(1:sps2:end);
            xx = xx3(1:Nc);

            rx_code_before = double(real(xx)<0);
            coded_ber_before_mc(mc) = mean(rx_code_before(num+1:Nc) ~= code_data(num+1:Nc));
            rx_msg_before = vitdec(rx_code_before,trellis,tblen,'trunc','hard');
            decoded_ber_before_mc(mc) = mean(rx_msg_before(num_info+1:N) ~= msg_source(num_info+1:N));
            xx_all(mc,:) = xx;
        end

        b = ss.';

        %% ---------- 294组参数 × 25次MC并行计算 ----------
        coded_flat     = nan(MC*numParam,1);
        decoded_flat   = nan(MC*numParam,1);
        mse_train_flat = nan(MC*numParam,1);
        mse_dec_flat   = nan(MC*numParam,1);

        parfor idx = 1:MC*numParam
            ip = mod(idx-1,numParam) + 1;
            mc = ceil(idx/numParam);

            delt = param_delt(ip);
            N1 = param_N1(ip);
            N2 = param_N2(ip);
            xT = xx_all(mc,:).';

            [y,~,~,er_panjue,mse_xulian,mse_panjue,~] = ...
                adaptDFEDPLL_NLMS(xT,b,N1,N2,num,delt,0.01,0.001,star);

            coded_flat(idx) = er_panjue/(Nc-num);

            rx_code_after = zeros(1,Nc);
            rx_code_after(1:num) = code_data(1:num);
            rx_code_after(num+1:Nc) = double(real(y(num+1:Nc))<0);
            rx_msg_after = vitdec(rx_code_after,trellis,tblen,'trunc','hard');
            decoded_flat(idx) = mean(rx_msg_after(num_info+1:N) ~= msg_source(num_info+1:N));

            mse_train_flat(idx) = mean(mse_xulian);
            if isempty(mse_panjue)
                mse_dec_flat(idx) = NaN;
            else
                mse_dec_flat(idx) = mean(mse_panjue);
            end
        end

        %% ---------- 还原294 × MC矩阵 ----------
        coded_ber_after_mc   = reshape(coded_flat,numParam,MC);
        decoded_ber_after_mc = reshape(decoded_flat,numParam,MC);
        mse_train_mc         = reshape(mse_train_flat,numParam,MC);
        mse_decision_mc      = reshape(mse_dec_flat,numParam,MC);

        %% ---------- 当前信道294组统计 ----------
        mean_codedBER = mean(coded_ber_after_mc,2,'omitnan');
        std_codedBER = std(coded_ber_after_mc,0,2,'omitnan');
        mean_decBER = mean(decoded_ber_after_mc,2,'omitnan');
        std_decBER = std(decoded_ber_after_mc,0,2,'omitnan');
        median_decBER = median(decoded_ber_after_mc,2,'omitnan');
        worst_decBER = max(decoded_ber_after_mc,[],2,'omitnan');
        mean_trainMSE = mean(mse_train_mc,2,'omitnan');
        mean_decisionMSE = mean(mse_decision_mc,2,'omitnan');
        complexity = param_N1 + param_N2 + 1;

        resultTable = table( ...
            (1:numParam).',param_delt,param_N1,param_N2,complexity, ...
            mean_codedBER,std_codedBER,mean_decBER,std_decBER, ...
            median_decBER,worst_decBER,mean_trainMSE,mean_decisionMSE, ...
            'VariableNames',{ ...
            'ParamIndex','delt','N1','N2','Complexity', ...
            'MeanCodedBER','StdCodedBER','MeanDecodedBER','StdDecodedBER', ...
            'MedianDecodedBER','WorstDecodedBER','MeanTrainMSE','MeanDecisionMSE'});

        %% ---------- 与原始代码相同的最优参数排序 ----------
        resultTableSorted = sortrows(resultTable, ...
            {'MeanCodedBER','MeanDecodedBER','StdDecodedBER','MeanDecisionMSE','Complexity'}, ...
            {'ascend','ascend','ascend','ascend','ascend'});

        bestResult = resultTableSorted(1,:);
        bestIndex = bestResult.ParamIndex;

        resultTable.GapToBestCodedBER = resultTable.MeanCodedBER - bestResult.MeanCodedBER;
        resultTable.GapToBestDecBER = resultTable.MeanDecodedBER - bestResult.MeanDecodedBER;

        best_mc_coded = coded_ber_after_mc(bestIndex,:);
        best_mc_dec = decoded_ber_after_mc(bestIndex,:);
        resultTable.WinRateCodedVsBest = zeros(numParam,1);
        resultTable.WinRateVsBest = zeros(numParam,1);

        for ip = 1:numParam
            resultTable.WinRateCodedVsBest(ip) = mean(coded_ber_after_mc(ip,:) <= best_mc_coded);
            resultTable.WinRateVsBest(ip) = mean(decoded_ber_after_mc(ip,:) <= best_mc_dec);
        end

        resultTableSorted = sortrows(resultTable, ...
            {'MeanCodedBER','MeanDecodedBER','StdDecodedBER','MeanDecisionMSE','Complexity'}, ...
            {'ascend','ascend','ascend','ascend','ascend'});
        bestResult = resultTableSorted(1,:);

        %% ---------- 只写入总结构体，不单独保存当前信道 ----------
        all_grid_results(kk).H = H;
        all_grid_results(kk).coded_ber_before_mc = coded_ber_before_mc;
        all_grid_results(kk).decoded_ber_before_mc = decoded_ber_before_mc;
        all_grid_results(kk).coded_ber_after_mc = coded_ber_after_mc;
        all_grid_results(kk).decoded_ber_after_mc = decoded_ber_after_mc;
        all_grid_results(kk).mse_train_mc = mse_train_mc;
        all_grid_results(kk).mse_decision_mc = mse_decision_mc;

        % 显式保存294组参数本身，便于和294×MC矩阵逐行对应
        all_grid_results(kk).param_delt = param_delt;
        all_grid_results(kk).param_N1 = param_N1;
        all_grid_results(kk).param_N2 = param_N2;

        % resultTable：当前信道全部294组参数的统计结果（不是只保存最优）
        all_grid_results(kk).resultTable = resultTable;

        % resultTableSorted：当前信道全部294组参数排序后的统计结果
        all_grid_results(kk).resultTableSorted = resultTableSorted;

        % bestResult只是额外保存最优那一行，绝不替代上面294组完整结果
        all_grid_results(kk).bestResult = bestResult;
        all_grid_results(kk).simulation_success = true;

        %% ---------- 把当前信道全部294组统计结果追加到总表 ----------
        ChannelIndexCol = repmat(kk,numParam,1);
        ChannelNameCol = repmat(string(currName),numParam,1);
        ChannelPathCol = repmat(string(filePath),numParam,1);
        SNR_dB_Col = repmat(currSNR,numParam,1);

        currentParamTable = addvars(resultTable, ...
            ChannelIndexCol,ChannelNameCol,ChannelPathCol,SNR_dB_Col, ...
            'Before',1, ...
            'NewVariableNames',{'ChannelIndex','ChannelName','ChannelPath','SNR_dB'});

        all_param_results = [all_param_results; currentParamTable]; %#ok<AGROW>

        fprintf('当前信道已保存全部 %d 组参数的详细统计结果。\n',height(resultTable));
        fprintf('当前信道294×MC原始矩阵尺寸：%d × %d。\n',size(coded_ber_after_mc,1),size(coded_ber_after_mc,2));

        fprintf('\n当前信道最优参数：\n');
        disp(bestResult);
        fprintf('前10组参数：\n');
        disp(resultTableSorted(1:min(10,height(resultTableSorted)),:));
        fprintf('均衡前：平均编码BER = %.6g，平均译码BER = %.6g\n', ...
            mean(coded_ber_before_mc,'omitnan'),mean(decoded_ber_before_mc,'omitnan'));

    catch ME
        all_grid_results(kk).simulation_success = false;
        all_grid_results(kk).error_message = ME.message;
        fprintf(2,'当前信道仿真失败：%s\n',ME.message);
    end

    %% ---------- 仍只保存同一个MAT，作为断点进度 ----------
    completedChannels = kk;
    save(allResultMatFile, ...
        'all_grid_results','all_param_results','completedChannels', ...
        'channel_names','channel_paths','snr_db', ...
        'msg_source','code_data','ss','MC','num','num_info', ...
        'delt_set','N1_set','N2_set','param_delt','param_N1','param_N2', ...
        'M','Rb','Rs','fc','fs','fss','beta','span','train_ratio','-v7.3');
    fprintf('当前进度已写入统一MAT：%d / %d\n',kk,numFiles);
end

%% ====================== 汇总所有信道最优结果 ======================
ChannelName = strings(numFiles,1);
ChannelPath = strings(numFiles,1);
SNR_dB = nan(numFiles,1);
Best_delt = nan(numFiles,1);
Best_N1 = nan(numFiles,1);
Best_N2 = nan(numFiles,1);
Best_Complexity = nan(numFiles,1);
Best_MeanCodedBER = nan(numFiles,1);
Best_StdCodedBER = nan(numFiles,1);
Best_MeanDecodedBER = nan(numFiles,1);
CodedBER_BeforeEQ = nan(numFiles,1);
DecodedBER_BeforeEQ = nan(numFiles,1);
SimulationSuccess = false(numFiles,1);
ErrorMessage = strings(numFiles,1);

for kk = 1:numFiles
    ChannelName(kk) = string(all_grid_results(kk).channel_name);
    ChannelPath(kk) = string(all_grid_results(kk).channel_path);
    SNR_dB(kk) = all_grid_results(kk).snr_db;
    SimulationSuccess(kk) = all_grid_results(kk).simulation_success;
    ErrorMessage(kk) = string(all_grid_results(kk).error_message);

    if all_grid_results(kk).simulation_success
        br = all_grid_results(kk).bestResult;
        Best_delt(kk) = br.delt;
        Best_N1(kk) = br.N1;
        Best_N2(kk) = br.N2;
        Best_Complexity(kk) = br.Complexity;
        Best_MeanCodedBER(kk) = br.MeanCodedBER;
        Best_StdCodedBER(kk) = br.StdCodedBER;
        Best_MeanDecodedBER(kk) = br.MeanDecodedBER;
        CodedBER_BeforeEQ(kk) = mean(all_grid_results(kk).coded_ber_before_mc,'omitnan');
        DecodedBER_BeforeEQ(kk) = mean(all_grid_results(kk).decoded_ber_before_mc,'omitnan');
    end
end

summaryTable = table(ChannelName,ChannelPath,SNR_dB, ...
    Best_delt,Best_N1,Best_N2,Best_Complexity, ...
    Best_MeanCodedBER,Best_StdCodedBER,Best_MeanDecodedBER, ...
    CodedBER_BeforeEQ,DecodedBER_BeforeEQ,SimulationSuccess,ErrorMessage);

%% ====================== 最终仍只保存一个MAT文件 ======================
completedChannels = numFiles;
save(allResultMatFile, ...
    'all_grid_results','all_param_results','summaryTable','completedChannels', ...
    'channel_names','channel_paths','snr_db', ...
    'msg_source','code_data','ss','MC','num','num_info', ...
    'delt_set','N1_set','N2_set','param_delt','param_N1','param_N2', ...
    'M','Rb','Rs','fc','fs','fss','beta','span','train_ratio','-v7.3');

fprintf('\n============================================================\n');
fprintf('全部信道294组网格仿真完成！\n');
fprintf('信道总数：%d\n',numFiles);
fprintf('成功：%d\n',sum(SimulationSuccess));
fprintf('失败：%d\n',sum(~SimulationSuccess));
fprintf('all_param_results 总行数 = %d（每个成功信道294行）\n',height(all_param_results));
fprintf('最终只保存一个MAT文件：\n%s\n',allResultMatFile);
fprintf('MAT中 all_grid_results(kk) 保留每个信道的294×%d原始MC矩阵和完整294组表。\n',MC);
toc

%% ========================================================================
%% 以下 adaptDFEDPLL_NLMS 函数保持原始代码不变
%% ========================================================================

function [y ,e ,est_w ,er_panjue,mse_xulian,mse_panjue,mse_theta] = adaptDFEDPLL_NLMS( xT,b ,N1,N2,num,mu,K1,K2,star)
%要了解原理，lxp前馈滤波器是后端信号对当前影响，反馈滤波器是前端信号对当前影响
%前馈的抽头为-K1至0，反馈为1至K2，参考数字通信
%判决反馈
%xT为输入数据 
%b为参考数据  
%Ne为均衡器长度  
%num为训练序列长度
%delt为步长 0.02
% Ne1 = 4*(Ne-1)/5;
% Ne2=(Ne-1)/5;
% Ne1 = (Ne-1)/5;
% Ne2=4*(Ne-1)/5;
% Ne1 = round(2*(Ne)/3);
% Ne2=round((Ne)/3);
Ne1 = N1;
Ne2 = N2;
N=length(b);
% xx=mapminmax(xT')';
w=zeros(Ne1+Ne2,N+1);   %lxp抽头系数，前馈8个，后馈7个
I=zeros(Ne2,1);   %判决序列，训练模式期间为已知序列，判决模式期间为实际判决序列

% star=[1i 1 -1i -1];
mse_xulian=[];
mse_panjue=[];
theta=0;
p=[];
q=[];
mse_theta=[];
xT = [zeros(round(Ne1/2),1);xT];
%lxp前馈滤波器是后端信号对当前影响，反馈滤波器是前端信号对当前影响，下面公式不太匹配
%如果b为参考序列，那么训练序列与参考序列个数不相同
% for i=1:num  
%     I=b(i+Ne2-1:-1:i);
%     y(i)=w(:,i)'*[xT(Ne1+i+Ne2:-1:i+Ne2);I];
%     err(i)=b(Ne1+i)-y(i);
%     e(i)=(abs(err(i)))^2;
%     w(:,i+1)=w(:,i)+delt*conj(err(i))*[xT(Ne1+i+Ne2:-1:i+Ne2);I];%采用LMS准则 为何共轭在err处？
%  %   w(:,i+1)=w(:,i)+delt*err(i)*conj([xT(Ne1+i+Ne2:-1:i+Ne2);I]);%采用LMS准则
%     mse_xulian = [mse_xulian e(i)];  %lxp
% end
%%

for i=1:num   %lxp训练序列应该与参考数据长度一致，即b数量与num一致，b前面补0
   % I=b(i+Ne2-1:-1:i);
    in=[xT(Ne1-1+i:-1:i)*exp(-j*theta);I];
    y(i)=w(:,i)'*[xT(Ne1-1+i:-1:i)*exp(-j*theta);I];
    p(i)=w(1:Ne1,i)'*[xT(Ne1-1+i:-1:i)*exp(-j*theta)];
    q(i)=w(Ne1+1:end,i)'*I;
    err(i)=b(i)-y(i);
    e(i)=(abs(err(i)))^2;
    w(:,i+1)=w(:,i)+mu*conj(err(i))*in/(in'*in);%采用LMS准则 为何共轭在err处？
    fai(i)=imag(p(i)*conj((b(i)-q(i))));
    theta=theta+K1*fai(i)+K2*sum(fai(1:i));
    mse_theta=[mse_theta,theta];
    I=[b(i);I(1:end-1)];
    mse_xulian = [mse_xulian e(i)];  %lxp
end
%%
%  figure;
% plot((mse_xulian)); grid on; %lxp
% title('mse_xulian');
% delt1 = 0.001;
er_panjue=0;
xT=[xT;zeros(Ne1,1)]; %%补够0
for i=num+1:N  %与上面更改的对应
%for i=num:N-Ne1  %lxp因为此时的I对应的是第num点的，所以这里重新在num上开始循环
    in=[xT(Ne1-1+i:-1:i)*exp(-j*theta);I];
    y(i)=w(:,i)'*[xT(Ne1-1+i:-1:i)*exp(-j*theta);I];
    p(i)=w(1:Ne1,i)'*[xT(Ne1-1+i:-1:i)*exp(-j*theta)];
    q(i)=w(Ne1+1:end,i)'*I;
    distance=abs(star-y(i));
    [~,index]=min(distance);
    err(i)=star(index)-y(i);
    e(i)=(abs(err(i)))^2;
    w(:,i+1)=w(:,i)+mu*conj(err(i))*in/(in'*in);
 %   w(:,i+1)=w(:,i)+delt*err(i)*conj([xT(Ne1+i:-1:i);I]);  %lxp加了conj
    fai(i)=imag(p(i)*conj((star(index)-q(i))));
    theta=theta+K1*fai(i)+K2*sum(fai(1:i));
    I=[star(index);I(1:end-1)];
    if(abs(star(index)-b(i))>0.001) 
        er_panjue=er_panjue+1;
    end
     mse_panjue = [mse_panjue e(i)];  %lxp
     mse_theta=[mse_theta,theta];
end
est_w = w;  %lxp选取最后的抽头系数
% figure;
% plot(mse_panjue);  %lxp
% title('mse_panjue');
% figure;
% plot(mse_theta);  %lxp
% title('mse_theta');
end


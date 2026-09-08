clear all;
close all;
clc;
tic

rng(22);                                  % 固定随机种子，便于复现
% parpool('local');                       % 如需要可手动开启并行池

%% ====================== 读取预测结果 ======================
% batch_predictions.mat 中已经包含：
% channel_names、channel_paths、snr_db、mu、N1、N2、
% candidate_id、success 等信息
%
% 不再读取 fileSNR.mat
% 不再使用 rate = all_results.rate
% 不再使用 ber  = all_results.ber

predictionMatFile = 'D:\机器学习\均衡器参数选择\批量信道预测\结果\20260906_212914_134057\batch_predictions.mat';
pred = load(predictionMatFile);

channel_names = pred.channel_names;
channel_paths = pred.channel_paths;
snr_db        = pred.snr_db;
mu_pred       = pred.mu;
N1_pred       = pred.N1;
N2_pred       = pred.N2;

if isfield(pred,'candidate_id')
    candidate_id = pred.candidate_id;
else
    candidate_id = nan(size(snr_db));
end

if isfield(pred,'success')
    prediction_success = logical(pred.success);
else
    prediction_success = true(size(snr_db));
end

% 预测模型给出的 score，如果存在则一起保存
prediction_score = nan(size(snr_db));
if isfield(pred,'results') && isstruct(pred.results) && isfield(pred.results,'score')
    try
        prediction_score = reshape([pred.results.score],[],1);
    catch
        prediction_score = nan(size(snr_db));
    end
end

numFiles = numel(channel_paths);

%% ====================== 统一结果保存位置 ======================
% 所有信道的仿真结果最后只保存到这一个 MAT 文件
saveDir = 'D:\matlab_code\Batch_BPSK\prediction_sim_results\';

if ~exist(saveDir,'dir')
    mkdir(saveDir);
end

allResultMatFile = fullfile(saveDir,'batch_prediction_simulation_results.mat');

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

MC = 25;                            % 每个信道/预测参数组合的Monte Carlo次数

% 信道编码
L = 7;
tblen = 3*L;
trellis = poly2trellis(7,[133 171]);    % (2,1,7)卷积编码

% 发送数据长度
N = 200;                            % 原始信息比特数

% 训练比例
train_ratio = 0.4;

%% ====================== 固定发送序列 ======================
% 所有信道、所有MC均使用同一套发送数据
msg_source = randi([0 M-1],1,N);
code_data = convenc(msg_source,trellis);
Nc = length(code_data);                    % 1/2码率：Nc = 2N

aa = code_data;
ss = exp(1j*aa*pi);
star = [1+0*1i,-1+0*1i];

num = round(train_ratio*Nc);                % 编码符号训练长度
num = 2*floor(num/2);
num_info = num/2;

%% ====================== 发射机 ======================
up_ss_ch = upsample(ss,sps2);
rcos_fir = rcosdesign(beta,span,sps2);
rcos_ss_ch = conv(up_ss_ch,rcos_fir,'same');

t = (1:length(rcos_ss_ch))/fs;
tx_signal = rcos_ss_ch.*exp(1j*2*pi*fc.*t);
tx_signal = real(tx_signal);

% 接收机固定低通滤波器
fir_lp = fir1(128,0.2);

%% ====================== 初始化总结果 ======================
simulation_results = repmat(struct( ...
    'channel_name','', ...
    'channel_path','', ...
    'snr_db',NaN, ...
    'mu',NaN, ...
    'N1',NaN, ...
    'N2',NaN, ...
    'candidate_id',NaN, ...
    'prediction_score',NaN, ...
    'prediction_success',false, ...
    'simulation_success',false, ...
    'error_message','', ...
    'H',[], ...
    'coded_ber_before_mc',[], ...
    'decoded_ber_before_mc',[], ...
    'coded_ber_after_mc',[], ...
    'decoded_ber_after_mc',[], ...
    'mse_train_mc',[], ...
    'mse_decision_mc',[], ...
    'mean_codedBER_before',NaN, ...
    'std_codedBER_before',NaN, ...
    'mean_decodedBER_before',NaN, ...
    'std_decodedBER_before',NaN, ...
    'mean_codedBER_after',NaN, ...
    'std_codedBER_after',NaN, ...
    'mean_decodedBER_after',NaN, ...
    'std_decodedBER_after',NaN, ...
    'mean_trainMSE',NaN, ...
    'mean_decisionMSE',NaN), numFiles, 1);

%% ====================== 批量读取预测参数并进行通信仿真 ======================
for kk = 1:numFiles

    fprintf('\n============================================================\n');
    fprintf('处理第 %d / %d 个信道\n',kk,numFiles);

    %% ---------- 从 batch_predictions.mat 读取当前条目 ----------
    if iscell(channel_paths)
        filePath = char(channel_paths{kk});
    else
        filePath = char(string(channel_paths(kk)));
    end

    if iscell(channel_names)
        currName = char(channel_names{kk});
    else
        currName = char(string(channel_names(kk)));
    end

    currSNR = double(snr_db(kk));
    delt    = double(mu_pred(kk));
    N1      = round(double(N1_pred(kk)));
    N2      = round(double(N2_pred(kk)));

    currCandidateID = double(candidate_id(kk));
    currPredSuccess = logical(prediction_success(kk));
    currPredScore   = double(prediction_score(kk));

    fprintf('信道：%s\n',currName);
    fprintf('SNR = %.4f dB\n',currSNR);
    fprintf('预测参数：mu = %.6f, N1 = %d, N2 = %d\n',delt,N1,N2);
    fprintf('candidate_id = %.0f\n',currCandidateID);

    % 先把预测信息写入总结果
    simulation_results(kk).channel_name = currName;
    simulation_results(kk).channel_path = filePath;
    simulation_results(kk).snr_db = currSNR;
    simulation_results(kk).mu = delt;
    simulation_results(kk).N1 = N1;
    simulation_results(kk).N2 = N2;
    simulation_results(kk).candidate_id = currCandidateID;
    simulation_results(kk).prediction_score = currPredScore;
    simulation_results(kk).prediction_success = currPredSuccess;

    % 如果预测本身失败，则跳过当前条目
    if ~currPredSuccess
        simulation_results(kk).error_message = 'batch_predictions.mat 中该条 prediction success = false';
        fprintf('该条预测标记为失败，跳过通信仿真。\n');
        continue;
    end

    try
        %% ====================== 读取当前信道 h_vary ======================
        % batch_predictions.mat 保存的是信道路径和预测参数，
        % 实际 h_vary 仍从 channel_path 指向的原始信道 MAT 文件读取。
        channelData = load(filePath,'h_vary');

        if ~isfield(channelData,'h_vary')
            error('当前信道文件中不存在 h_vary 变量。');
        end

        h_vary = channelData.h_vary;

        %% ====================== 原始信道处理逻辑 ======================
        Index0_center = 626;                % 主径位置
        win_side = round(2*fss);
        startIdx = Index0_center - win_side;
        endIdx   = min(size(h_vary,2), Index0_center + win_side);
        h_vary = h_vary(:,startIdx:endIdx);

        H = h_vary(end,:);
        H_pd = shift_H_pd(H,fs,fc);

        Index0 = find(H_pd == max(H_pd));
        NNa = find(H_pd == max(H_pd)) - 1;  % 信道的非因果长度
        NNc = size(H_pd,2) - Index0;         % 信道的因果长度

        h_vary_pd = shift_H_pd_all(h_vary(626:end,:),fs,fc,fss);

        %% ====================== 发射信号通过当前动态信道 ======================
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

        %% ====================== 当前信道MC结果矩阵 ======================
        coded_ber_before_mc   = nan(1,MC);
        decoded_ber_before_mc = nan(1,MC);

        coded_ber_after_mc    = nan(1,MC);
        decoded_ber_after_mc  = nan(1,MC);

        mse_train_mc          = nan(1,MC);
        mse_decision_mc       = nan(1,MC);

        b = ss.';

        %% ====================== Monte Carlo ======================
        % 当前信道只测试 batch_predictions.mat 给出的这一组参数
        % 不再进行294组网格搜索
        parfor mc = 1:MC

            % ---------- 加噪 ----------
            ch_p = awgn(rx_clean,currSNR,'measured');

            % ---------- 接收机前端 ----------
            xx1 = ch_p.*exp(-1j*2*pi*fc.*t);
            rcos_ss_1p = conv(xx1,fir_lp,'same');
            xx2 = conv(rcos_ss_1p,rcos_fir,'same');
            xx3 = xx2(1:sps2:end);
            xx = xx3(1:Nc);

            % ---------- 均衡前 ----------
            rx_code_before = double(real(xx)<0);

            coded_ber_before_mc(mc) = mean( ...
                rx_code_before(num+1:Nc) ~= code_data(num+1:Nc));

            rx_msg_before = vitdec(rx_code_before,trellis,tblen,'trunc','hard');

            decoded_ber_before_mc(mc) = mean( ...
                rx_msg_before(num_info+1:N) ~= msg_source(num_info+1:N));

            % ---------- NLMS-DFE-DPLL ----------
            xT = xx.';

            [y,~,~,er_panjue,mse_xulian,mse_panjue,~] = ...
                adaptDFEDPLL_NLMS(xT,b,N1,N2,num,delt,0.01,0.001,star);

            % ---------- 均衡后编码BER ----------
            coded_ber_after_mc(mc) = er_panjue/(Nc-num);

            % ---------- 均衡后Viterbi译码 ----------
            rx_code_after = zeros(1,Nc);
            rx_code_after(1:num) = code_data(1:num);
            rx_code_after(num+1:Nc) = double(real(y(num+1:Nc))<0);

            rx_msg_after = vitdec(rx_code_after,trellis,tblen,'trunc','hard');

            decoded_ber_after_mc(mc) = mean( ...
                rx_msg_after(num_info+1:N) ~= msg_source(num_info+1:N));

            % ---------- MSE ----------
            mse_train_mc(mc) = mean(mse_xulian);

            if isempty(mse_panjue)
                mse_decision_mc(mc) = NaN;
            else
                mse_decision_mc(mc) = mean(mse_panjue);
            end
        end

        %% ====================== 当前信道统计 ======================
        mean_codedBER_before   = mean(coded_ber_before_mc,'omitnan');
        std_codedBER_before    = std(coded_ber_before_mc,0,'omitnan');

        mean_decodedBER_before = mean(decoded_ber_before_mc,'omitnan');
        std_decodedBER_before  = std(decoded_ber_before_mc,0,'omitnan');

        mean_codedBER_after    = mean(coded_ber_after_mc,'omitnan');
        std_codedBER_after     = std(coded_ber_after_mc,0,'omitnan');

        mean_decodedBER_after  = mean(decoded_ber_after_mc,'omitnan');
        std_decodedBER_after   = std(decoded_ber_after_mc,0,'omitnan');

        mean_trainMSE          = mean(mse_train_mc,'omitnan');
        mean_decisionMSE       = mean(mse_decision_mc,'omitnan');

        %% ====================== 写入统一结果结构体 ======================
        simulation_results(kk).H = H;

        simulation_results(kk).coded_ber_before_mc = coded_ber_before_mc;
        simulation_results(kk).decoded_ber_before_mc = decoded_ber_before_mc;
        simulation_results(kk).coded_ber_after_mc = coded_ber_after_mc;
        simulation_results(kk).decoded_ber_after_mc = decoded_ber_after_mc;
        simulation_results(kk).mse_train_mc = mse_train_mc;
        simulation_results(kk).mse_decision_mc = mse_decision_mc;

        simulation_results(kk).mean_codedBER_before = mean_codedBER_before;
        simulation_results(kk).std_codedBER_before = std_codedBER_before;
        simulation_results(kk).mean_decodedBER_before = mean_decodedBER_before;
        simulation_results(kk).std_decodedBER_before = std_decodedBER_before;

        simulation_results(kk).mean_codedBER_after = mean_codedBER_after;
        simulation_results(kk).std_codedBER_after = std_codedBER_after;
        simulation_results(kk).mean_decodedBER_after = mean_decodedBER_after;
        simulation_results(kk).std_decodedBER_after = std_decodedBER_after;

        simulation_results(kk).mean_trainMSE = mean_trainMSE;
        simulation_results(kk).mean_decisionMSE = mean_decisionMSE;
        simulation_results(kk).simulation_success = true;

        %% ====================== 命令行显示 ======================
        fprintf('均衡前：编码BER = %.6g，译码BER = %.6g\n', ...
            mean_codedBER_before,mean_decodedBER_before);

        fprintf('均衡后：编码BER = %.6g，译码BER = %.6g\n', ...
            mean_codedBER_after,mean_decodedBER_after);

        fprintf('训练MSE = %.6g，判决MSE = %.6g\n', ...
            mean_trainMSE,mean_decisionMSE);

    catch ME
        simulation_results(kk).simulation_success = false;
        simulation_results(kk).error_message = ME.message;

        fprintf(2,'当前信道仿真失败：%s\n',ME.message);
    end

end

%% ====================== 生成总汇总表 ======================
ChannelName = strings(numFiles,1);
ChannelPath = strings(numFiles,1);

SNR_dB = nan(numFiles,1);
mu = nan(numFiles,1);
N1 = nan(numFiles,1);
N2 = nan(numFiles,1);
CandidateID = nan(numFiles,1);
PredictionScore = nan(numFiles,1);

PredictionSuccess = false(numFiles,1);
SimulationSuccess = false(numFiles,1);

CodedBER_BeforeEQ = nan(numFiles,1);
DecodedBER_BeforeEQ = nan(numFiles,1);
CodedBER_AfterEQ = nan(numFiles,1);
StdCodedBER_AfterEQ = nan(numFiles,1);
DecodedBER_AfterEQ = nan(numFiles,1);
StdDecodedBER_AfterEQ = nan(numFiles,1);
TrainMSE = nan(numFiles,1);
DecisionMSE = nan(numFiles,1);

for kk = 1:numFiles
    ChannelName(kk) = string(simulation_results(kk).channel_name);
    ChannelPath(kk) = string(simulation_results(kk).channel_path);

    SNR_dB(kk) = simulation_results(kk).snr_db;
    mu(kk) = simulation_results(kk).mu;
    N1(kk) = simulation_results(kk).N1;
    N2(kk) = simulation_results(kk).N2;
    CandidateID(kk) = simulation_results(kk).candidate_id;
    PredictionScore(kk) = simulation_results(kk).prediction_score;

    PredictionSuccess(kk) = simulation_results(kk).prediction_success;
    SimulationSuccess(kk) = simulation_results(kk).simulation_success;

    CodedBER_BeforeEQ(kk) = simulation_results(kk).mean_codedBER_before;
    DecodedBER_BeforeEQ(kk) = simulation_results(kk).mean_decodedBER_before;

    CodedBER_AfterEQ(kk) = simulation_results(kk).mean_codedBER_after;
    StdCodedBER_AfterEQ(kk) = simulation_results(kk).std_codedBER_after;

    DecodedBER_AfterEQ(kk) = simulation_results(kk).mean_decodedBER_after;
    StdDecodedBER_AfterEQ(kk) = simulation_results(kk).std_decodedBER_after;

    TrainMSE(kk) = simulation_results(kk).mean_trainMSE;
    DecisionMSE(kk) = simulation_results(kk).mean_decisionMSE;
end

summaryTable = table( ...
    ChannelName,ChannelPath,SNR_dB,mu,N1,N2,CandidateID,PredictionScore, ...
    PredictionSuccess,SimulationSuccess, ...
    CodedBER_BeforeEQ,DecodedBER_BeforeEQ, ...
    CodedBER_AfterEQ,StdCodedBER_AfterEQ, ...
    DecodedBER_AfterEQ,StdDecodedBER_AfterEQ, ...
    TrainMSE,DecisionMSE);

%% ====================== 只保存一个总 MAT 文件 ======================
% 不再为每个信道单独保存 grid_xxx.mat
% 不再为每个信道单独保存 CSV
% 不再保存 summary.xlsx
%
% simulation_results：每个信道的详细MC结果
% summaryTable：所有信道的汇总结果
% prediction_input：原 batch_predictions.mat 中的预测信息

prediction_input = pred;

save(allResultMatFile, ...
    'simulation_results', ...
    'summaryTable', ...
    'prediction_input', ...
    'msg_source','code_data','ss', ...
    'MC','num','num_info', ...
    'M','Rb','Rs','fc','fs','fss','beta','span','train_ratio', ...
    '-v7.3');

fprintf('\n============================================================\n');
fprintf('全部仿真完成！\n');
fprintf('总信道数：%d\n',numFiles);
fprintf('成功仿真：%d\n',sum(SimulationSuccess));
fprintf('失败/跳过：%d\n',numFiles-sum(SimulationSuccess));
fprintf('所有结果只保存到一个 MAT 文件：\n%s\n',allResultMatFile);

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

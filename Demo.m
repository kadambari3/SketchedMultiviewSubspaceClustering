%%%% In this code, we will implement all the methods on linear dataset 

clc;clear all;close all


addpath('Performance measures')



D = 50; %Dimension of ambient space
n = 2; %Number of subspaces
d1 = 30; d2 = 20; %d1 and d2: dimension of subspace 1 and 2
N1 = 300; N2 = 400; %N1 and N2: number of points in subspace 1 and 2
X1 = randn(D,d1) * randn(d1,N1); %Generating N1 points in a d1 dim. subspace
X2 = randn(D,d2) * randn(d2,N2); %Generating N2 points in a d2 dim. subspace
Data1 = [X1 X2];


P1 = construct_random_mat(70, 50);
P2 = orth(randn(70));
Data2 = P2*P1*Data1;

X{1,1} = Data1;
X{2,1} = Data2;
gnd = [1*ones(1,N1) 2*ones(1,N2)]';
clear D n d1 d2 N1 N2 X1 X2 P1 P2 Data1 Data2


N = size(X{1,1},2); % Number of data samples
v = size(X,1); % number of views
nVal = [10]'; 

%%% LMvSSC
lam_LMvSSC = 1*ones(v,1);
alp_LMvSSC = 5500*ones(v,1);
r_LMvSSC = -0.21;
k_nn_LMvSSC = 7; 

%%% SRMvSSC
lam_SRMvSSC = 1*ones(v,1);
alp_SRMvSSC = 50*ones(v,1);
k_nn_SRMvSSC  = 10;
r_SRMvSSC = -5;

%%% KMvSSC
params_kernel.KernelType =  'gaussian';
params_kernel.sigma = [50 55]'; % view 1 and view 2
lam_KMvSSC = 1*ones(v,1);
alp_KMvSSC  = 50*ones(v,1);
r_KMvSSC = -5;
k_nn_KMvSSC = 5; % k-nn for initial graph construction


%%% Sketch LSR
lam_SketchLSR = 2000;
k_nn_SketchLSR = 15;


RandType = 'normal';

NIter = 1; % number of Iters for each nVal
MaxIter = 300; % maxIters for algorithm to converge
for ii = 1:length(nVal)
n = nVal(ii,1);

parfor jj = 1:NIter
    
    nv = 0.8;
    [Objec_LMvSSC(:,jj), acc_LMvSSC(ii,jj), fscore_LMvSSC(ii,jj), prec_LMvSSC(ii,jj), recall_LMvSSC(ii,jj),...
        nmi_LMvSSC(ii,jj), ar_LMvSSC(ii,jj), time_LMvSSC(ii, jj)] = Perf_LMvSSC(X, n, RandType, gnd, nv, k_nn_LMvSSC, r_LMvSSC , alp_LMvSSC, lam_LMvSSC, MaxIter);
    
    [Objec_SRMvSSC(:,jj), acc_SRMvSSC(ii,jj), fscore_SRMvSSC(ii,jj), prec_SRMvSSC(ii,jj), recall_SRMvSSC(ii,jj),...
        nmi_SRMvSSC(ii,jj), ar_SRMvSSC(ii,jj), time_SRMvSSC(ii, jj)] = Perf_SRMvSSC(X, n, r_SRMvSSC , gnd, nv, k_nn_SRMvSSC, alp_SRMvSSC, lam_SRMvSSC , MaxIter, 'normal');
                                                                         
    [Objec_KMvSSC(:,jj), acc_KMvSSC(ii,jj), fscore_KMvSSC(ii,jj), prec_KMvSSC(ii,jj), recall_KMvSSC(ii,jj),...
       nmi_KMvSSC(ii,jj), ar_KMvSSC(ii,jj), time_KMvSSC(ii, jj)] = Perf_KMvSSC(X, gnd, n, nv, k_nn_KMvSSC, RandType, r_KMvSSC, alp_KMvSSC, lam_KMvSSC, MaxIter, params_kernel);


end
end

rmpath('Performance measures')

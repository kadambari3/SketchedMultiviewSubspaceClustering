function [Objec, acc, fscore, prec, recall, nmi, ar, time] = Perf_KMvSSC(X, gnd, n, nv, k_nn, RandType, r, alp, lam, MaxIter, params_kernel);

[v] = size(X, 1); % number of views
N = size(X{1,1}, 2); % number of data points
Nc = max(unique(gnd)); % number of clusters

% Add noise and get kernel for each view
for i = 1:v
    
    Xv = X{i,1}; % data in ith view
    Nv = nv*rand(size(Xv)); % noise matrix
    Xn = Xv + nv*Nv;
    
    % Apply kernel
    sv = params_kernel.sigma(i,1);
    opts.KernelType = 'gaussian';
    opts.sigma = sv;
    Kv = construct_kernel(Xn', Xn', opts); % Kernel 
    
    
    X{i,1} = Xn; % noisy version of data
    K{i,1} = Kv; % kernel 
    
    clear Xv Nv Xn sv opts
end
clear i;


tic;
%%%%%% Generate random matrices for each view
R = cell(v,1); % place holder
for i = 1:v
    R{i,1} = construct_random_mat(N, n, RandType);
end
clear i


%%%%%% Main algorithm

% Place holders
Ac = zeros(n,N); % initalize centroid
Objec = zeros(MaxIter,1);
dist_Av = zeros(v, MaxIter);
dist_h = zeros(v, MaxIter);
Lam = zeros(v, MaxIter);


for k = 1:MaxIter
    
    
    for i =1:v
        Kv = K{i,1};
        Rv = R{i,1};
        lv = lam(i,1);
        av = alp(i,1);
        
        Z1 = lv*Rv'*Kv*Rv + av*eye(n);
        Z2 = av*Ac + lv*Rv'*Kv;
        opts_LinSolve.SYM = true;
        opts_LinSolve.POSDEF = true;
        A{i,1} = linsolve(Z1, Z2, opts_LinSolve);
        clear Kv Rv lv av Z1 Z2 i opts_LinSolve
    end
    
    
    % Step 2: Find Ac
    ZZ = zeros(n,N);
    for i = 1:v
        Av = A{i,1};
        av = alp(i,1);
        ZZ = ZZ + av*Av;
        clear Av av
    end
    Ac = ZZ/sum(alp);
    clear ZZ
    
    
    % Step 3: Find lam
    p = 1/(r-1);
    for i = 1:v
        Kv = K{i,1};
        Av = A{i,1};
        Rv = R{i,1};
        h = trace(Kv - Kv*Rv*Av - Av'*Rv'*Kv + Av'*Rv'*Kv*Rv*Av);
        lam(i,1) = (-h/r)^p;
    
        dist_Av(i,k) = norm(Ac-Av, 'fro')^2;
        dist_h(i,k) = h;
        Lam(i,k) = lam(i,1);
        clear Kv Av Rv h
    end
    clear i
    
    
    % Compute objective function value at each iter
    Objec(k,1) = lam'*dist_h(:,k) + alp'*dist_Av(:,k) + sum(lam.^r);
    
    % Check convergence
    if (k>1) & (abs(Objec(k) - Objec(k-1))/(abs(Objec(k-1,1)))) < 1e-8
        fprintf('Converged at %d \n', k)
        break
    end
    
    
    
end


% Compute clustering performace
Zc = normc(Ac);
param.k = k_nn;
G = gsp_nn_graph(Zc', param);
W = full(G.W);

figure(10);clf;spy(W)

[F] = Find_F(W, Nc);
idx = kmeans(F,Nc,'maxiter',1000,'replicates',20,'EmptyAction','singleton');

acc = compute_clus_acc(gnd,idx);
[fscore, prec, recall] = compute_fscore(gnd, idx);
nmi= compute_nmi(gnd, idx);
ar = rand_index(gnd, idx);


time = toc;

end

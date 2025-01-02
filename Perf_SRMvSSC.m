function [Objec, acc, fscore, prec, recall, nmi, ar, time] = Perf_SRMvSSC(X, n, r, gnd, nv, k_nn, alp, lam, MaxIter, RandType)


N = size(X{1,1},2); % number of data points
v = size(X,1);  % number of views
Nc = length(unique(gnd)); % number of clusters 
%%%% Implements Alg. 3 (SR-MvSSC) in the paper. 

% W = zeros(N,N); % Place holder
% for i =1:v
%     param.k = k_nn;
%     Xv = X{i,1};
%     Gv = gsp_nn_graph(Xv', param); % k-nn graph for each view
%     Wv = full(Gv.W);
%     W = W + Wv;
%     clear Xv Gv Wv param
% end
% W = W/v; % Average graph from all the views
% L = diag(sum(W)) - W;
% clear W




% Add noise to multi-view data (add noise to each view)
for i = 1:v 
    Xv = X{i,1};
    Xn = Xv + nv*rand(size(Xv));
    X{i,1} = Xn; % Noisy data
    clear Xv Nv Xn
end
clear i


tic;
% Sketch multi-view data
for i = 1:v
    Xv = X{i,1};
    Rv = construct_random_mat(N, n, RandType);
    B{i,1} = Xv*Rv; % Dictionary matrix for each view
    clear Xv Rv
end
clear i


% % Find initial graph for the main algorithm (avg. k-nn graph formed by the
% % multiview data)
% 
W = zeros(N,N); % Place holder
for i =1:v
    param.k = k_nn;
    Xv = X{i,1};
    Gv = gsp_nn_graph(Xv', param); % k-nn graph for each view
    Wv = full(Gv.W);
    W = W + Wv;
    clear Xv Gv Wv param
end
W = W/v; % Average graph from all the views
L = diag(sum(W)) - W;
clear W



%%%%%%%%% Main algorithm
Objec = zeros(MaxIter, 1);

for k = 1:MaxIter
    
    % Step 1: Updating A for each view (solving sylvester equation)
    
    for i = 1:v
        Xv = X{i,1};
        Bv = B{i,1};
        av = alp(i,1);
        lv = lam(i,1);
        
        Z1 = lv*Bv'*Bv;
        Z2 = av*L;
        Z3 = lv*Bv'*Xv;
        Av = sylvester(Z1, Z2, Z3);
        A{i,1} = Av;
        clear Xv Bv av lv Z1 Z2 Z3 Av
    end
    
    % Step 2: updating L (learn approx graph from data)
    
    % Compute pair-wise distance matrices
    Z = zeros(N,N);
    for i = 1:v
        Av = A{i,1};
        av = alp(i,1);
        % Compute pairwise distance
        Zv = gsp_distanz(Av).^2; % using gsp toolbox
        Z = Z + 0.5*av*Zv;
        clear Av av Zv
    end
    clear i
    
    % Learn approx. graph
    D = Z;
    [~, idx] = sort(D, 2); % sort each row
    S = zeros(N);
    for i = 1:N
        id = idx(i,2: k_nn+2);
        di = D(i, id);
        S(i,id) = (di(k_nn+1)-di)/(k_nn*di(k_nn+1)-sum(di(1:k_nn))+eps);
    end
    
    W = 0.5*(S + S');
    clear D Z idx S id di
    
    
    % Step 3: updating lam
    p = 1/(r-1);
    for i = 1:v
        Xv = X{i,1};
        Bv = B{i,1};
        Av = A{i,1};
        
        hv = norm(Xv - Bv*Av, 'fro')^2;
        lam(i,1) = (-hv/r)^(p);
        
        % params for computing objective function
        Rep_error(i,k) = hv;
        Smoothness(i,k) = trace(Av*L*Av');
        
        clear Xv Bv Av hv
    end
    clear p
    
    
    
    % Compute objective function
    Objec(k,1) =   Rep_error(:,k)'*lam + Smoothness(:,k)'*alp + sum(lam.^r);
    
    
    % convergence
    if (k>1) & abs(Objec(k,1) - Objec(k-1,1))/(abs(Objec(k-1,1))) < 1e-8
        fprintf('Converged at %d \n', k)
        break
    end
    
    
end

% plot W as a graph
% figure(2);clf;plot(graph(W))


% Performance on initial graph
[F] = Find_F(W, Nc);
idx = kmeans(F, Nc, 'maxiter',1000,'replicates',20,'EmptyAction','singleton');

acc = compute_clus_acc(gnd,idx);
[fscore, prec, recall] = compute_fscore(gnd, idx);
nmi = compute_nmi(gnd, idx);
ar = rand_index(gnd, idx);

time = toc;


end

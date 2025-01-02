function [Objec, acc, fscore, prec, recall, nmi, ar, time] = Perf_LMvSSC(X, n, RandType, gnd, nv, k_nn, r, alp, lam, MaxIter)
 





[v] = size(X, 1); % number of views
N = size(X{1,1}, 2); % number of data points
Nc = max(unique(gnd)); % number of clusters


% Add noise to multiview data
for i = 1:v
    X{i,1} = X{i,1} + nv*randn(size(X{i,1}));
end


tic;
%%%%%%%%%%%%%%% Generate random sketching matrix
R = cell(v,1); % place holder
for i = 1:v % Sketch data in each view
R = construct_random_mat(N, n, RandType);
B{i,1} = X{i,1}*R;
clear R
end
clear i



Ac = zeros(n,N); % initalize centroid
Objec = zeros(MaxIter,1);
for k = 1:MaxIter
    
    % Find Av for each view
    for i = 1:v
        Xv = X{i,1};
        Bv = B{i,1};    
        lv = lam(i,1);
        av = alp(i,1);
        
        
        Z1 = lv*Bv'*Bv + av*eye(n);
        Z2 = av*Ac + lv*Bv'*Xv;
        
%         opts.SYM = true;
%         opts.POSDEF = true;
%         A{i,1} = linsolve(Z1, Z2, opts);
        A{i,1} = Z1\Z2; % inv(Z1)*Z2;
        clear Kv Rv lv av Z1 Z2 i        
    end
    
    % Find Ac    
    ZZ = zeros(n,N);
    for i = 1:v
        Av = A{i,1};
        av = alp(i,1);
        ZZ = ZZ + av*Av;
        clear Av av
    end
    Ac = ZZ/sum(alp);
    clear ZZ
    
    %%%%%%% update lam 
    p = 1/(r-1);
    for i = 1:v
        Xv = X{i,1};
        Av = A{i,1};       
        Bv = B{i,1};
        
        h = norm(Xv - Bv*Av, 'fro')^2;
        lam(i,1) = (-h/r)^p;
        
        
        
        dist_Av(i,1) = norm(Ac-Av, 'fro')^2; 
        dist_h(i,1) = h;
        clear Kv Av Rv h
    end
    clear i
    
    
    % Compute objective function value at each iter
    Objec(k,1) = lam'*dist_h + alp'*dist_Av + sum(lam.^r);
    
    Objec';
     
    
    
    if (k>1) & abs(Objec(k) - Objec(k-1))/(abs(Objec(k-1,1))) < 1e-8
        fprintf('Converged at %d \n', k)
        break
    end
    
end

% Compute clustering performace
Zc = normc(Ac);




param.k = k_nn;
G = gsp_nn_graph(Zc', param);

[F] = Find_F(G.W, Nc);
idx = kmeans(F,Nc,'maxiter',1000,'replicates',20,'EmptyAction','singleton');

acc = compute_clus_acc(gnd,idx);
[fscore, prec, recall] = compute_fscore(gnd, idx);
nmi = compute_nmi(gnd, idx);
ar = rand_index(gnd, idx);

time = toc;

end
function [F] = Find_F(W, Nc)
N = size(W,2);

diff = eps;

% Normalize Laplacian matrix 
DN = sum(W,2);
%DN(find(DN)) = (1./sqrt(DN(find(DN))));

DN(DN~=0) = (1./sqrt(DN(DN~=0)));
%DN = diag( 1./sqrt(sum(W)+eps) );

DN=spdiags(DN,0,speye(size(W,1)));
LapN = speye(N) - DN * W * DN;
LapN = LapN + 0.001*speye(N); % Adding Identity matrix to handel numberical unstatility
[Vn,~] = eigs(LapN, Nc, diff);

% F = bsxfun(@rdivide, Vn, sqrt(sum(Vn.^2, 2))+eps);
F = DN*Vn;
end
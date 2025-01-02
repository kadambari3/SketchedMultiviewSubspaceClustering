function [A] = CGL_Find_A(X, B, L, lam, alp)
    
    v = size(X,1);
    N = size(X{1,1},2);
    
    
    for i = 1:v
        Xv = X{i,1};
        Bv = B{i,1};
        av = alp(i,1);
        lv = lam(i,1);
        
        Z1 = lv*Bv'*Bv;
        Z2 = av*L;
        Z3 = lv*Bv'*Xv;
        
        A{i,1} = sylvester(Z1, Z2, Z3);
        
        clear Xv Bv av lv Z1 Z2 Z3
    end
    end
function [X,T,objFun] = Gradient_Method(Alpha, Beta, Func, Jac, isFeasible,  X0, MaxIt)   
    %apply Gradient metho and return the function, X for each iteration 
    %along with the step 
    %@Alpha backtracking line search alpha 
    %@Beta backtracking line search beta
    %@Func is the input function(s) for which we are seeking the roots    
    %@Jac is a function to evaluate the Jacobian
    %@Hesse is a function to evaluate the Hessain
    %@X0 is the initial guess     
    %@MaxIt is the max number of iterations     
    x = X0;    
    X = x';
    n = 1;
    tol = 1e-10;%tolerance
    T = 1;
    objFun = Func(X0);
    while n < MaxIt
        %1) compute function and gradient    
        myF = Func(x);
        objFun = [objFun, myF];
        myJ = Jac(x);
        
        %2) stopping criterion based gradient norm 
        if norm(myJ) < tol
            break;
        end
                
        t = 1;
        %do line search while respecting the constraints 
        %find the right t 
        while ~isFeasible(x-t*myJ)
             t = Beta*t;
        end        
        while Func(x-t*myJ) > myF - Alpha*t*(myJ'*myJ)
            t = Beta*t;
        end 
        
        
        %4) update 
        x = x - t*myJ;
        T = [T,t];
        X = [X; x'];
        n=n+1;
    end
    
    
end
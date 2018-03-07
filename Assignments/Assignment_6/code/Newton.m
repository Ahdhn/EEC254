function [X,T,objFun] = Newton(Alpha, Beta, Func, Jac, Hesse , isFeasible,  X0, MaxIt)   
    %apply Newton method and return the function, X for each iteration  
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
        %1) compute newton step         
        myF = Func(x);
        objFun = [objFun, myF];
        myJ = Jac(x);        
        myH = Hesse(x);        
        sol = mldivide(-myH,myJ);

        %2) stopping criterion
        Lamda = myJ'*sol;
        if(Lamda*Lamda/2.0 < tol) %based on Newton decrement
        %if abs(normF(end)) < tol || abs(norm(X(end)) - norm(X(end-1))) <tol
            %based on the size of the norm and step size 
            break;
        end        
        
        %3) backtracking line search
        t = 1;
        %do line search while respecting the constraints 
        %find the right t 
        while ~isFeasible(x+t*sol)
             t = Beta*t;
        end        
        while Func(x+t*sol) > myF + Alpha*t*Lamda
            t = Beta*t;
        end 
        %4) update 
        x = x + t*sol;
        T = [T,t];
        X = [X; x'];
        n=n+1;
    end
end
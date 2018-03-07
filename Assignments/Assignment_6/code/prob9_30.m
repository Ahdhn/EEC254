clc;
clear;
close all;
disp('EEC 254 - HW6 - Problem 9.30:');
%%%%%%%%%%%%%%%%%%% Part a)
global A numRows numCols;
numRows = 1000;
numCols = 1000;
A = instance(numRows, numCols);%init matrix A
x = zeros(numCols,1);%init vector x
Alpha = 0.9;
Beta = 0.5;

[X_grad, steplen_grad, objFunc_grad] = Gradient_Method(Alpha,Beta, @Func, @Grad, @isFeasible, x, 50);
objFun_p_grad = objFunc_grad - objFunc_grad(end);

figure
semilogy(objFun_p_grad,'k', 'LineWidth',1.5);
xlabel('Iteration');
ylabel('f(x)^{(k)}-p^*');
label1 = 'Gradient Method';
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;

figure
plot(steplen_grad,'m','LineWidth',1.5);
xlabel('Iteration');
ylabel('Step Size');
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;

figure
plot(objFunc_grad,'r','LineWidth',1.5);
xlabel('Iteration');
ylabel('Objective Function');
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;

if length(X_grad(1,:)) == 2
    %plot the contours of the domain if its 2d 
    %plotNorm(1,X_grad,@Func);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X_newton, steplen_newton, objFunc_newton] = Newton(Alpha,Beta, @Func, @Grad, @Hessain,@isFeasible, x, 50);
objFun_p_newton = objFunc_newton - objFunc_newton(end);

if length(X_newton(1,:)) == 2
    %plot the contours of the domain if its 2d 
    %plotNorm(2,X_newton,@Func);
end

figure
semilogy(objFun_p_newton,'k', 'LineWidth',1.5);
xlabel('Iteration');
ylabel('f(x)^{(k)}-p^*');
label1 = 'Newton Method';
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;

figure
plot(steplen_newton,'m','LineWidth',1.5);
xlabel('Iteration');
ylabel('Step Size');
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;

figure
plot(objFunc_newton,'r','LineWidth',1.5);
xlabel('Iteration');
ylabel('Objective Function');
lgd = legend(label1,'Location','best');
lgd.FontSize = 12;


disp('Done!!');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Instance %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = instance(numRows, numCols)
    %creating a problem instance 
    A = rand(numRows,numCols);    
    A=A./(2.0*max(max(A)));    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%Plot Contours %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotNorm(fignum, X, func)
    v=[10.3,.01:1:900];
    xr=-0.99:0.005:0.99;
    n=length(xr);
    z=zeros(n,n);
    for i=1:n
        for j=1:n        
            z(i,j)=func([xr(i),xr(j)]);
        end
    end
    figure(fignum);
    contourf(xr,xr,z);
    ylim([-1 1]);
    xlim([-1 1]);
    hold 
    
    plot(X(:,1),X(:,2),'-*');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function myVal = Func(x)
    global A numRows numCols
    %Evluate the function on x
    %x is a vector of length numCols 
    %A is numRows x numCols
    myVal = 0;
    for i=1:numRows
        myVal = myVal + log(1- dot(A(i,:),x));
    end
    for i=1:numCols
        myVal = myVal + log(1- x(i)*x(i));
    end    
    myVal = -myVal;
end
function myGrad = Grad(x)
    global A
    %Evluate the gradient on x    
    %using chain rule
    myGrad = A'*(1./(1-A*x)) - 1./(1+x) + 1./(1-x);
end
function myHessain = Hessain(x)
    global A    
    %Evluate the hessain on x    
    %using chain rule
    myHessain = A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2); 
    
end
function fe = isFeasible(x)
    global A    
    %check if x is in the feasible region by checking the constraints 
    fe = true;
    if max(A*x) >= 1.0 %= to avoid numerical issues 
        fe = false;
    end
    
    if max(abs(x)) >= 1.0 
        fe = false;
    end

end
clear;
clc;

n = 32;
%min area constraints (uniform)
MinArea = 100*ones(1,n)';
%Max and min aspect ratio
omega_max = 2.0*ones(n,1);
omega_min = 0.5*ones(n,1);
%generate relative positions constraints 
[graph_L, graph_U] = genRelativePos(MinArea);

%Do the optimization
cvx_begin quiet
    variables x(n) y(n);
    variable w(n) nonnegative;
    variable h(n) nonnegative;
    variable W nonnegative;
    variable H nonnegative;
    minimize 2*(H+W)
    subject to
        0 <= x <= W-w;
        0 <= y <= H-h;
        diag(x)*graph_L + diag(w)*graph_L - graph_L*diag(x) <= 0;
        diag(y)*graph_U + diag(h)*graph_U - graph_U*diag(y) <= 0;
        MinArea .* inv_pos(h) - w <=0;
        omega_min .* w <= h;
        h  <=  omega_max .* w;
cvx_end

%compute area     
area_covered = 0;
for i=1:n
    area_covered = area_covered + w(i)*h(i);        
end
total_area = (max(x+w)-min(x))*(max(y+h)-min(y));
waste_percent =   100*(total_area - area_covered)/total_area
%This part is borrowed from CVX examples to draw the cells nicely     
fill([0; W; W; 0],[0;0;H;H],[1 1 1]);
hold on
for i=1:n
    fill([x(i); x(i)+w(i); x(i)+w(i); x(i)] , [y(i); y(i) ;y(i)+h(i);y(i)+h(i)], 0.90*[1 1 1]);
    hold on;
    text(x(i)+w(i)/2, y(i)+h(i)/2,int2str(i));
end
axis([0 W 0 H]);
axis equal;
axis off;



function [groupA, groupB] = randomPartition(areas)
    %Parition the areas array into two groups such that the sum of 
    %areas in each group in equal 
    sumAreaA = 0;
    sumAreaB = 0;
    groupA = [];
    groupB = [];
    randomPerm = randperm(length(areas));    
    for i = randomPerm    
        if sumAreaA < sumAreaB
            groupA = [groupA i];            
            sumAreaA = sumAreaA + areas(randomPerm(i));            
        else
            groupB = [groupB i];            
            sumAreaB = sumAreaB + areas(randomPerm(i));            
        end         
    end    
end 
function [graph_L, graph_U] = genRelativePos(areas)
    %generate the relative positions graph/matrix using the min areas array
    graph_L = zeros(length(areas));
    graph_U = zeros(length(areas));
    [graph_L, graph_U] = genRelativePos_recursive(areas, 1:length(areas),...
                                                  graph_L, graph_U,0);    
end
function [graph_L, graph_U] = genRelativePos_recursive(areas, index, graph_L, graph_U, direction)
    %recursivly generate the relative positions graphs  
    
    [a1, a2] = randomPartition(areas);       
    
    for i = index(a1)        
        for j = index(a2)            
            if direction == 0  
                graph_L(i,j) = 1;
            else
                graph_U(i,j) = 1;
            end 
        end
    end
    
    if length(a1) >1 
        [graph_L, graph_U] = genRelativePos_recursive(areas(a1),index(a1),...
                             graph_L, graph_U, ~direction);
    end
    if length(a2)>1
        [graph_L, graph_U] = genRelativePos_recursive(areas(a2),index(a2),...
                             graph_L, graph_U, ~direction);
    end       
    
end
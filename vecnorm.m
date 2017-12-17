function [res] = vecnorm(vec, p, dim)
    N = size(vec,2)
    tmp = zeros(1,N);
    for i=1:N
        tmp(i) = norm(vec(:,i),2);
    end
    res = tmp;
end

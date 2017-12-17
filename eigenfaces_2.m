function [eigenvectors, eigenvalues, mean_face] = eigenfaces_2(X)
    N = size(X, 2);
    mean_face = mean(X, 2);
    
    % 4. substract mean face
    A = X - mean_face;
    
    % 5. Covariance matrix
    S = 1 / N * A' * A;
    
    % 6. Compute eigen values (doesn't actually take that long to compute)
    [eigenvectors, eigenvalues] = eig(S);
end
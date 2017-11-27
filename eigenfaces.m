function [eigenvectors, eigenvalues, mean_face] = eigenfaces(X)
    N = size(X, 2);
    mean_face = 1 / N * mean(X, 2);
    
    % 4. substract mean face
    A = X - mean_face;
    size(mean_face) 
    
    % 5. Covariance matrix
    S = 1.0 / N * A * A';
    
    % 6. Compute eigen values (doesn't actually take that long to compute)
    [eigenvectors, eigenvalues] = eig(S);
    
    % Since it takes awhile to compute, save them somewhere
    save("eigenfaces_1.mat", "eigenvectors", "eigenvalues", "mean_face");
end
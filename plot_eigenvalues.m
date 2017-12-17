function [] = plot_eigenvalues(eigenvalues)
    d = diag(eigenvalues);
    d = d(1:364);
    
    figure;
    
    plot(d);
    grid;
    title("Eigenvalues of entire training data (364 images)");
    ylabel("Eigenvalue");
    hold on;
    print("eigenfaces_plots/eigenvalues_slow_all", "-deps");
    
    figure;
    plot(d(1:50));
    grid;
    title("Eigenvalues of first 50 principle components");
    print("eigenfaces_plots/eigenvalues_slow_50", "-deps");
    ylabel("Eigenvalue");
    
    figure;
    v = cumsum(d) / sum(d);
    plot(v)
    grid;
    title("Components vs variance retained");
    print("eigenfaces_plots/eigenvectors_variance", "-deps");
    
    
end
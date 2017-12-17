
[W,lambda] = eig(inv(S_w_pca)*S_b_pca);

X_train_pca = U' * A;

means_pca = zeros(N,num_eigen_bases);

for i = 1:N
   idx = 1+(i-1)*train_samp;
   for j=1:num_eigen_bases
       for k=1:train_samp
          contrib = (W(j,:)*X_train_pca(:,idx-1+k));
          mean_project_contrib = contrib / train_samp;
          means_pca(i,j) = means_pca(i,j) + mean_project_contrib;
       end
   end
end

X_train_lda = W' * X_train_pca;

X_test_pca = ((X_test - meanface2)'*U)';
X_validation_pca = ((X_validation' - meanface2)'*U)';

test_size = size(X_test_pca,2);

y_test_guess = zeros(1, test_size);
y_validation_guess = zeros(1, 52);
validation_size = 52;

for i=1:validation_size
    proj = W(:,1:lda_dim)'*X_validation_pca(:,i);
    dists = zeros(1,52 * 7);
    
%     for j = 1:N;
%         dists(j) = pdist2(means_pca(j,1:lda_dim),proj');
%     end
    
    for t = 1:size(X_train_lda, 2)
      d = (X_train_lda(1:lda_dim, t) - proj)';
      dists(:, t) = sum(d .^ 2);
    end
    
    
    [dist, bla] = min(dists);
    y_validation_guess(i) = y_train(bla);
end

for i=1:test_size
    proj = W(:,1:lda_dim)'*X_test_pca(:,i);
    dists = zeros(1,52 * 7);
    
%     for j = 1:N;
%         dists(j) = pdist2(means_pca(j,1:lda_dim),proj');
%     end
    
    for t = 1:size(X_train_lda, 2)
      d = (X_train_lda(1:lda_dim, t) - proj)';
      dists(:, t) = sum(d .^ 2);
    end
    
    
    [dist, bla] = min(dists);
    y_test_guess(i) = y_train(bla);
end

fprintf(" - lda_dim = %d validation = %.3f, test = %.3f\n", ...
    lda_dim, ...
    sum(y_validation_guess == y_validation) / length(y_validation_guess), ...
    sum(y_test_guess == y_test) / length(y_test_guess));
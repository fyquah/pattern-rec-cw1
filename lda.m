load data_partition.mat


N = 52;
train_samp = 7;
test_samp = 3;
feature_len = size(X_train,1);
means = zeros(N,feature_len);
X_t = X_train';

S_i = cell(N,1);
S_w = zeros(feature_len,feature_len);

for i = 1:N
   idx = 1+(i-1)*train_samp;
   means(i,:) = mean(X_t(idx:idx+6,:));
   S_i(i) = {zeros(feature_len,feature_len)};
   for j = 1:train_samp
        x_del = X_t(idx+j-1,:)-means(i,:);
        S_x =  x_del'*x_del;
        S_i(i) = {S_i{i} + S_x};
   end
   S_w = S_w + S_i{i};
end

meanface = mean(means);

S_b = zeros(feature_len,feature_len);


for i = 1:N
    meanx_del = means(i, :) - meanface;
    S_b = S_b + meanx_del'*meanx_del;
end


% PCA stuff
[evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
A = X_train - meanface2;
normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);
U = normalized_evectors2(:, 1:51);

S_w_pca = U'*S_w*U;
S_b_pca = U'*S_b*U;


lda_dim = 7;

[W,lambda] = eig(inv(S_w_pca)*S_b_pca);

X_train_pca = (A'*U)';

means_pca = zeros(N,51);

for i = 1:N
   idx = 1+(i-1)*train_samp;
   for j=1:51
       for k=1:train_samp
          mean_project_contrib = (W(j,:)*X_train_pca(:,idx-1+k))/train_samp;
          means_pca(i,j) = means_pca(i,j) + mean_project_contrib;
       end
   end
end

X_test_pca = ((X_test - meanface2)'*U)';

test_size = size(X_test_pca,2)

y_guess = zeros(1,test_size)

for i=1:test_size
    proj = W(:,1:lda_dim)'*X_test_pca(:,i);
    dists = zeros(1,52);
    for j = 1:N;
        dists(j) = pdist2(means_pca(j,1:lda_dim),proj');
    end
    [dist, y_guess(i)] = min(dists);
end



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


choices = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 51];

for lda_dim = 11:20
    the_real_lda  
end
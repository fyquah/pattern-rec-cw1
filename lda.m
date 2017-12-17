load data_partition.mat

N = 52;
train_samp = 7;
test_samp = 3;
feature_len = size(X_train,1);
means = zeros(train_samp,feature_len);
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
    meanx_del = means(i) - meanface;
    S_b = S_b + meanx_del'*meanx_del;
end


[W,lambda] = eig(inv(S_w)*S_b);

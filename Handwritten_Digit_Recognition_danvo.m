%MAT 167 - Final Project
%Dannie Vo - Student ID: 915004803
%Step 1a
%load the USPS file into Matlab
load USPS.mat;
% if the jth handwritten digit in train patterns represents i, then the (i + 1, j)th entry
% of train labels is +1, else it is -1 for all the other entries that are not i+1.
for i = 0:9
    for j = 1:train_patterns
        if train_patterns(:,j) == i
            train_labels(i+1,j) = +1;
        else
            l = ~(i+1);
            train_labels(l,j) = -1;
        end
    end
end
%Step 1b
%Create a new figure window (1) and color it gray
figure;
colormap(gray);
%Print out the first 16 images of train_patterns using subplot() and
%imagesc() function
for k = 1:16
    subplot(4,4,k);
    imagesc(reshape(train_patterns(:,k), 16, 16)');
end

%Step 2
%Create a new figure window (2) and color it gray
figure;
colormap(gray);

%Create train_aves matrix with size 256x10, representing the mean digits in
%train_patterns
%Print the 10 mean digits images using subplot() and imagesc() function
train_aves = [];
for k = 1:10
    k_avg = mean(train_patterns(:,train_labels(k,:)==1),2);
    subplot(2,5,k);
    imagesc(reshape(k_avg, 16,16)');
    train_aves = [train_aves k_avg];
end 

%Step 3a
%Create a 10 x 4649 matrix called test_classif that shows the Euclidean
%distance from test_patterns and each mean digit from train_aves
test_classif = [];
for k = 1:10
     k_sum = sum((test_patterns-repmat(train_aves(:,k),[1 4649])).^2);
     test_classif(k,:) = k_sum;
end

%Step 3b
%Create a vector of size 1 x 4649 called test_classif_res that shows the
%index od the minimum of each column in test_classif, where the index are
%from 1 to 0
test_classif_res = [];
for j = 1:4649
    [tmp, ind] = min(test_classif(:,j));
    test_classif_res(:,j) = ind;
end

%Step 3c
%Create a 10 x 10 matrix called test_confusion to calculate how many
%entries that have that value for each k- 1st digit
test_confusion = [];
for k = 1:10
    tmp=test_classif_res(test_labels(k,:)==1);
    for j = 1:10
        test_confusion(k,j) = sum(tmp==j);
    end
end
test_confusion
%Write the test_confusion matrix under latex
test_confusion_latex = latex(sym(test_confusion));

%Step 4a
%Create the train_u array of size 256x17x10 with 17 singular values and
%vectors after pooling all images that correspond to the kth digit of
%train_patterns
for k = 1:10
    [train_u(:,:,k),tmp,tmp2] = svds(train_patterns(:,train_labels(k,:)==1),17);
end

%Step 4b
%Create a 17x4649x10 3D array called test_svd17 that shows the expansion
%coefficients of each test digit image.
for k = 1:10
    test_svd17(:,:,k) = train_u(:,:,k)' * test_patterns;
end

%Step 4c
%Create a 10x4649 matrix called test_svd17res that shows the error between
%the test image patterns and its rank 17 approximation using kth digit
%images in the training data
test_svd17res = [];
for k = 1:10
    rank17_appro = train_u(:,:,k)*test_svd17(:,:,k);
    svd_appro = test_patterns - rank17_appro;
    test_svd17res(k,:) = sum(svd_appro(:,:).^2);
end

%Step 4d
%Create a 1 x 4649 matrix that helps us find the position index of the
%minimun of each column of the test_svd17res matrix
test_svd17res_a = [];
for j = 1:4649
    [tmp, ind] = min(test_svd17res(:,j));
    test_svd17res_a(:,j) = ind;
end
%Create a 10 x 10 matrix called test_svd17_confusion to calculate how many
%entries that have that value for each k- 1st digit
test_svd17_confusion = [];
for k = 1:10
    tmp=test_svd17res_a(test_labels(k,:)==1);
    for j = 1:10
        test_svd17_confusion(k,j) = sum(tmp==j);
    end
end
test_svd17_confusion
%Write the test_svd17_confusion matrix into latex
test_svd17_confusion_latex = latex(sym(test_svd17_confusion));

%Step 5c
%Using the simplest algorithm
%Calculate the percentage of each digit from test_confusion 
for k = 1:10
    test_conf_perc_each_digit(k,:) = [k-1 (test_confusion(k,k)/sum(test_confusion(k,:)))*100];
end
%test_conf_perc_each_digit matrix has 2 columns where the first column is
%the digit and the second column is the corresponding percentage of each
%digit being identified
test_conf_perc_each_digit
latex_A = latex(vpa(sym(test_conf_perc_each_digit),6));
%Using the k-nearest neighbor classification algorithm
%Calculate the percentage of each digit from test_svd17_confusion
for k = 1:10
    test_svd17_conf_perc_each_digit(k,:) = [k-1 (test_svd17_confusion(k,k)/sum(test_svd17_confusion(k,:)))*100];
end
%test_svd17_conf_perc_each_digit matrix has 2 columns where the first column is
%the digit and the second column is the corresponding percentage of each
%digit being identified
test_svd17_conf_perc_each_digit
latex_B = latex(vpa(sym(test_svd17_conf_perc_each_digit),6));

%Step 5d
%Calculate the percentage of the test_confusion and test_svd17_confusion to
%see which algorithm is more effective
sum_A = 0;
for k = 1:10
    sum_A = sum_A + test_confusion(k,k);
end
%The total percentage from the simplest algorithm using test_confusion
%matrix
percentage_test_confusion = (sum_A / 4649)*100
sum_B = 0;
for k = 1:10
    sum_B = sum_B + test_svd17_confusion(k,k);
end
%The total percentage from the k-nearest neighbor classification algorithm
%using test_svd17_confusion
percentage_test_svd17_confusion = (sum_B / 4649)*100

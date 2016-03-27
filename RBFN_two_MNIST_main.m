%% Homework 3 question 2
clear all
close all
clc

%load MNIST_dataset.mat;
% This code assumes that you have the famous MNIST dataset in your working
% directory.
% Adapt the following line according to your settings
%addpath('C:\Users\User\Dropbox\Dropbox\Machine Learning Projects\ML_summer\matconvnet-master\data\mnist')
addpath('/home/rob/Dropbox/Machine Learning Projects/ML_summer/matconvnet-master/data/mnist');

%Credits for the use of these functions go to Andrew Ng and his coursera
%course
train_data = loadMNISTImages('train-images-idx3-ubyte');
train_classlabel = loadMNISTLabels('train-labels-idx1-ubyte');
test_data = loadMNISTImages('t10k-images-idx3-ubyte');
test_classlabel = loadMNISTLabels('t10k-labels-idx1-ubyte');

N = size(train_data,2);
Ntest = size(test_data,2);

columns = randperm(N,16);
for i = 1:16
    subplot(4,4,i)
    tmp=reshape(train_data(:,columns(i)),28,28);
    imshow(double(tmp));
end

% Select the assigned labels
% Target to a logical
trainIdx = find(train_classlabel==1| train_classlabel==8); % find the location of classes 0, 1, 2
y_train = [double(train_classlabel(trainIdx)==8)]';
X_train = [train_data(:,trainIdx)]';

testIdx = find(test_classlabel==1| test_classlabel==8); % find the location of classes 0, 1, 2
y_test = [double(test_classlabel(testIdx)==8)]';
X_test = [test_data(:,testIdx)]';

% Note that we use the full 60.000 MNIST dataset and select a subset of
% that
N_select = 250;  %How many train samples do you want?
N_test_select = 80;    %How many test samples do you want?
N = size(X_train,1);
Ntest = size(X_test,1);

idx = randperm(N,N_select);
X_train = X_train(idx,:);
y_train = y_train(idx);

idx = randperm(Ntest,N_test_select);
X_test = X_test(idx,:);
y_test = y_test(idx);
%
N = size(X_train,1);
Ntest = size(X_test,1);
D = size(X_train,2);

y_train = y_train(:);
y_test = y_test(:);

%% Start the RBFN
% For exact interpolation
cp_exact = X_train;
num_hidden_neuron = size(cp_exact,1);
train_input_dim = size(X_train,1);
test_input_dim = size(X_test,1);
var = 100^2;

phi_train = zeros(train_input_dim,num_hidden_neuron);
phi_test = zeros(test_input_dim,num_hidden_neuron);

for i = 1:train_input_dim
    for j = 1:num_hidden_neuron
        r = norm(X_train(i,:) - cp_exact(j,:));
        phi_train(i,j) = exp(-r^2/(2*var));
    end
end
for i = 1:test_input_dim
    for j = 1:num_hidden_neuron
        r = norm(X_test(i,:) - cp_exact(j,:));
        phi_test(i,j) = exp(-r^2/(2*var));
    end
end

%Select columns for approximate
num_hidden_neuron_app = 30;
ind = randperm(num_hidden_neuron,num_hidden_neuron_app);

phi_train_app = phi_train(:,ind);
phi_test_app = phi_test(:,ind);

%Solve the linear problem
w_exact = phi_train\y_train;
w_app = phi_train_app\y_train;

% Make predictions
y_train_pred = phi_train*w_exact;
y_test_pred = phi_test*w_exact;

y_train_pred_app = phi_train_app*w_app;
y_test_pred_app = phi_test_app*w_app;

% Evaluate
% First row is exact
% Second row is approx
TrAcc=zeros(2,1000);
TeAcc=zeros(2,1000);
thr=zeros(1,1000);
for i=1:1000
    t=(max(y_train_pred)-min(y_train_pred))*(i-1)/1000+min(y_train_pred) ;
    thr(i)=t;
    %Exact
    neg = find(y_train_pred<t);
    pos = find(y_train_pred>=t);
    TrAcc (1,i)= length(find(y_train(neg)==0))+length(find(y_train(pos)==1));
    TrAcc (1,i)= TrAcc(1,i)/length(y_train);
    neg = find(y_test_pred<t);
    pos = find(y_test_pred>=t);
    TeAcc (1,i)= length(find(y_test(neg)==0))+length(find(y_test(pos)==1));
    TeAcc (1,i)= TeAcc (1,i)/length(y_test);
    %Approx
    neg = find(y_train_pred_app<t);
    pos = find(y_train_pred_app>=t);
    TrAcc (2,i)= length(find(y_train(neg)==0))+length(find(y_train(pos)==1));
    TrAcc (2,i)= TrAcc(2,i)/length(y_train);
    neg = find(y_test_pred_app<t);
    pos = find(y_test_pred_app>=t);
    TeAcc (2,i)= length(find(y_test(neg)==0))+length(find(y_test(pos)==1));
    TeAcc (2,i)= TeAcc (2,i)/length(y_test);
end
subplot(2,1,1)
plot(thr, TrAcc(1,:),'.-',thr, TeAcc(1,:),'^-');
ylabel('Accuracy')
xlabel('Threshold')
legend('train accuracy','test accuracy')
title('Exact interpolation')

subplot(2,1,2)
plot(thr, TrAcc(2,:),'.-',thr, TeAcc(2,:),'^-');
ylabel('Accuracy')
xlabel('Threshold')
legend('train accuracy','test accuracy')
title('Approximate interpolation')

%% Vary the regularization
close all
clc
% Insert below which regularizations you are curious for
lambdas = [0.000001;0.3;0.5;1;1.5;3;10;30];
figure
for n = 1:length(lambdas)  %Loops over lambdas
    %construct the pseudo-inverse with regularization
    lambda = lambdas(n);
    proj = inv(phi_train'*phi_train+lambda*eye(num_hidden_neuron))*phi_train';
    w_reg = proj*y_train;
    
    %Make predictions
    y_train_pred_reg = phi_train*w_reg;
    y_test_pred_reg = phi_test*w_reg;
    
    % Evaluate
    % First row is exact
    % Second row is approx
    TrAcc=zeros(2,1000);
    TeAcc=zeros(2,1000);
    thr=zeros(1,1000);
    for i=1:1000
        t=(max(y_train_pred_reg)-min(y_train_pred_reg))*(i-1)/1000+min(y_train_pred_reg) ;
        thr(i)=t;
        %Exact
        neg = find(y_train_pred_reg<t);
        pos = find(y_train_pred_reg>=t);
        TrAcc (1,i)= length(find(y_train(neg)==0))+length(find(y_train(pos)==1));
        TrAcc (1,i)= TrAcc(1,i)/length(y_train);
        neg = find(y_test_pred_reg<t);
        pos = find(y_test_pred_reg>=t);
        TeAcc (1,i)= length(find(y_test(neg)==0))+length(find(y_test(pos)==1));
        TeAcc (1,i)= TeAcc (1,i)/length(y_test);
    end
    subplot(2,4,n)
    plot(thr, TrAcc(1,:),'.-',thr, TeAcc(1,:),'^-');
    ylabel('Accuracy')
    xlabel('Threshold')
    legend('train accuracy','test accuracy')
    title(sprintf('lamda %0.2e Best %0.2f',lambda,max(TeAcc(1,:))))
end

%% Vary the standard deviation
close all
clc
%insert below which standard deviations you are curious for
devs = 10.^[-5;0;1;1.5;2;2.5;3;4];
figure
for n = 1:length(devs)  %Loops over deviations
    dev = devs(n);
    %Construct the phi-matrices
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cp_exact = X_train;
    num_hidden_neuron = size(cp_exact,1);
    train_input_dim = size(X_train,1);
    test_input_dim = size(X_test,1);
    var = dev^2;
    
    phi_train = zeros(train_input_dim,num_hidden_neuron);
    phi_test = zeros(test_input_dim,num_hidden_neuron);
    
    for i = 1:train_input_dim
        for j = 1:num_hidden_neuron
            r = norm(X_train(i,:) - cp_exact(j,:));
            phi_train(i,j) = exp(-r^2/(2*var));
        end
    end
    for i = 1:test_input_dim
        for j = 1:num_hidden_neuron
            r = norm(X_test(i,:) - cp_exact(j,:));
            phi_test(i,j) = exp(-r^2/(2*var));
        end
    end
    
    %Select columns for approximate
    num_hidden_neuron_app = 100;
    ind = randperm(num_hidden_neuron,num_hidden_neuron_app);
    
    % [ones(size(phi_train,1),1)
    % [ones(size(phi_test,1),1)
    
    phi_train_app =   phi_train(:,ind);
    phi_test_app =  phi_test(:,ind);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %construct the pseudo-inverse
    
    proj = inv(phi_train_app'*phi_train_app)*phi_train_app';
    w_reg = proj*y_train;
    
    %Make predictions
    y_train_pred_app = phi_train_app*w_reg;
    y_test_pred_app = phi_test_app*w_reg;
    
    % Evaluate
    % First row is exact
    % Second row is approx
    TrAcc=zeros(2,1000);
    TeAcc=zeros(2,1000);
    thr=zeros(1,1000);
    for i=1:1000
        t=(max(y_train_pred_app)-min(y_train_pred_app))*(i-1)/1000+min(y_train_pred_app) ;
        thr(i)=t;
        %Exact
        neg = find(y_train_pred_app<t);
        pos = find(y_train_pred_app>=t);
        TrAcc (1,i)= length(find(y_train(neg)==0))+length(find(y_train(pos)==1));
        TrAcc (1,i)= TrAcc(1,i)/length(y_train);
        neg = find(y_test_pred_app<t);
        pos = find(y_test_pred_app>=t);
        TeAcc (1,i)= length(find(y_test(neg)==0))+length(find(y_test(pos)==1));
        TeAcc (1,i)= TeAcc (1,i)/length(y_test);
    end
    subplot(2,4,n)
    plot(thr, TrAcc(1,:),'.-',thr, TeAcc(1,:),'^-');
    ylabel('Accuracy')
    xlabel('Threshold')
    legend('train accuracy','test accuracy')
    title(sprintf('st dev %0.2e Best %0.2f',dev,max(TeAcc(1,:))))
end
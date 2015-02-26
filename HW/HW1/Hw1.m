% Homework 1 for machine learning 
% shuai zhang
% time: Jan 14th
clc; clear all; close all;

%% P1: load data 
iris=load('data/iris.txt'); % load the text file
y = iris(:,end); % target value is last column
X = iris(:,1:end-1); % features are other columns
whos % show current variables in memory and sizes

m = size(iris,1); n = size(iris,2);
mea = zeros(1,n); varr = mea; sd = varr;
ndata = zeros(m,n);
for i = 1:n
    % figure;
    % hist(iris(:,i));
    mea(i) = mean(iris(:,i));
    varr(i) = var(iris(:,i));
    sd(i) = std(iris(:,i));
    ndata(:,i) = (iris(:,i) - mea(i))/sd(i);
end

% i=1; 
% for j=2:4,
%     figure;
%     ids=find(y==0); plot(X(ids,i),X(ids,j),'b.','markersize',20); hold on;
%     ids=find(y==1); plot(X(ids,i),X(ids,j),'g.','markersize',20);
%     ids=find(y==2); plot(X(ids,i),X(ids,j),'r.','markersize',20); hold off;
% end


%% P2: kNN predictions
% separate data as training and test 
iris=load('data/iris.txt'); y=iris(:,end); X=iris(:,1:2);
[X y] = shuffleData(X,y);
% shuffle data randomly
% (This is a good idea in case your data are ordered in some pathological way, as the Iris data are)
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

K = [1,5,10,50];
for i = 1:length(K)
    figure;
    k = K(i)
    knn = knnClassify( Xtr, Ytr, k);
    YteHat = predict( knn, Xte );
    plotClassify2D( knn, Xtr, Ytr ); % make 2D classification plot with data (Xtr,Ytr)
end









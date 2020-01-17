% PRINCIPAL COMPONENETS ANALYSIS (PCA)
% [X]-mean(x))/Variance(X)
% Difference b/c PCA and SVD: measurements are rows, X = [x1;x2;...,x50]
% rows of V* play same role in PCA as solumns of U before

% Ovarian cancer data
% look as 3 most dominant modes (3 first cols of V)

clear all, close all, clc

% Load ovarian cancer data, MATLAB built-in data
% get a huge matrix called obs with 216 patients, 4000 genes (features) stacked in rows
% grp: group patients into cancer and normal groups
load ovariancancer; 

% Take svd, economy is the faster way
[U,S,V] = svd(obs,'econ');
figure
% 216 singular values
subplot(1,2,1)
semilogy(diag(S),'k-o','LineWidth',1.5)
set(gca,'FontSize',13), axis tight, grid on
% cumulative sum of 216 singular values
subplot(1,2,2)
plot(cumsum(diag(S))./sum(diag(S)),'k-o','LineWidth',1.5)
set(gca,'FontSize',13), axis tight, grid on

% plot each patitent in terms of 3 PCAs
figure, hold on
for i=1:size(obs,1)
    x = V(:,1)'*obs(i,:)';
    y = V(:,2)'*obs(i,:)';
    z = V(:,3)'*obs(i,:)';
    % cancer group in red x
    if(grp{i}=='Cancer')
    plot3(x,y,z,'rx','LineWidth',2);
    else
    % noncancer group in blue circle
    plot3(x,y,z,'bo','LineWidth',2);
    end
end
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
view(85,25), grid on, set(gca,'FontSize',13)
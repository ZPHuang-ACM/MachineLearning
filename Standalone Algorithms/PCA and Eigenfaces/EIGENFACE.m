clear all; close all; clc
% 255,255,255 = pure white; 0,0,0 = pure black
% https://introcomputing.org/image-6-grayscale.html

% Size of each picture in pixels
m = 200; % row
n = 175; % column

% Number of sample pictures
N = 20; % use 20 pics for each person

avg = zeros(m*n,1);  % the average face, big column vector
A = []; % store everything in a big matrix A

%% Load Arnold Schwarzenegger's pics
count = 0;
% N pictures
for j = 1:N
    figure(1)
    % Note %02d is required since the images are ordered as 01;02;...
    ff = ['faces/arnold',num2str(j,'%02d'),'.jpg'];
    % ff = ['faces/stallone',num2str(j,'%02d'),'.jpg'];
    u = imread(ff); % Read the image into a matrix
    % show the image
    imshow(u)
    if(size(u,3)==1) % if it is already gray
        M=double(u); 
    % for colored image, each pixel(point), has RGB three elements in [0,255] (unit8)
    else % if not, convert to gray scale image
        M=double(rgb2gray(u)); 
    end
    pause(0.1);
    % reshape into a big column vector
    R = reshape(M,m*n,1);
    % Stack the column vector into A
    A = [A,R];
    % keep adding R into the avg colunmn vector
    avg = avg + R;
   count = count + 1;
end

%% Load Sylvester Stallone
for j = 1:N
    figure(1)
    ff = ['faces/stallone',num2str(j,'%02d'),'.jpg'];
    % ff = ['faces/taylor',num2str(j,'%02d'),'.jpg'];
    u = imread(ff); % Read the image into a matrix
    imshow(u)
    % only get the gray scales
    M = double(u(:,:,1));
    % reshape the image
    R = reshape(M,m*n,1);
    A = [A,R];
    pause(0.1);
    avg = avg + R;
   count = count + 1;
end
avg = avg /count;
% 

%% Calculate the "averaged" face
% reshape the avgcolumn vector to a matrix and convert it to [0,255] so that we
% can plot as an image
avgTS = uint8(reshape(avg,m,n));
figure(1)
% avg faces of arnold and stallone
imshow(avgTS);

%% Center the sample pictures at the "origin"
% subtract the mean face, remove mean
figure(1)
for j = 1:2*N % j from 1 to 40
    A(:,j) = A(:,j) - avg; % subtract the avg in every column
    R = reshape(A(:,j),m,n); % reshape the zero-meaned faces
    imshow(R);   
    pause(0.1);
end
% 
%%  Computing the SVD
% economy SVD, much faster
[U,S,V] = svd(A,'econ');

% extract the 40 most important ones
Phi = U(:,1:2*N);
Phi(:,1) = -1*Phi(:,1); % why invert the sign? 

% Plot 9 most impostant eigenfaces
figure(2)
count = 1;
for i=1:3
    for j=1:3
        subplot(3,3,count)
        % here 25000 is just a scale so that we can plot using [0,255] scale
        imshow(uint8(25000*reshape(Phi(:,count),m,n)));
        count = count + 1;
    end
end
% 
% 
%% project each image onto basis (project onto three eigenfaces)
% Get a coordinate, just like PCA, 
% Just a coordinate transformation
% every image of arnold
for j = 1:N
    imvec = A(:,j);
    ARN(:,j) = imvec'*Phi(:,1:3); % 
end
% every image of stallone
for j = 1:N
    imvec = A(:,N+j);
    STAL(:,j) = imvec'*Phi(:,1:3);
end

% plot the three dominant eigenvector-transfromed coordinates
% shows certain clustering, like the ovarian cancer data set
figure(3)
% Arnold
plot3(ARN(1,:),ARN(2,:),ARN(3,:),'r.','MarkerSize',30)
hold on
% Stallone
plot3(STAL(1,:),STAL(2,:),STAL(3,:),'b.','MarkerSize',30)
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
legend('Arnold','Stallone')

%% add some unexpected pics
% To test to whether it can be classfied as Arnold or Stallone
u = imread('faces/teststallone1.jpg');        
figure(4)
subplot(1,2,1)
imshow(u);
% convert to gray scale
u = double(rgb2gray(u));
% reshape to column vector and remove mean
ustal = reshape(u,m*n,1)-avg;
% Coordinate transformation in PCA 
stalpts = ustal'*Phi(:,1:3);
% 
v = imread('faces/testterminator8.jpg');
subplot(1,2,2)
imshow(v);
% gray scale
v = double(rgb2gray(v));
% reshape to column vector and remove mean
vterm = reshape(v,m*n,1)-avg;
% Coordinate transformation in PCA, projection onto 3 PCAs
termpts = vterm'*Phi(:,1:3);
%%
figure(3)
% Stallone in green
plot3(stalpts(1),stalpts(2),stalpts(3),'g.','MarkerSize',30)
% Terminate in black
plot3(termpts(1),termpts(2),termpts(3),'k.','MarkerSize',30)





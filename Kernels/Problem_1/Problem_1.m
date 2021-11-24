%% Erotima 1

clc;
clear;
close all;

rng('default')
x=rand(1,1000); 
b=0.1; 

figure(1)
subplot(2,1,1)
[f,xi] = ksdensity(x,'NumPoints',1000, 'Bandwidth' ,b,'Kernel' , 'myGaussianKernel'); 
plot(xi,f);
hold on 
pd1 = makedist('Uniform'); %%Plot Uniform Distribution in red          
pdf1 = pdf(pd1,xi);
plot(xi,pdf1,'r'); 
title('Gaussian Kernel h=0.1 b=0.1') 

subplot(2,1,2)
[f,xi] = ksdensity(x,'NumPoints',1000,'Bandwidth' , b ,'Kernel', 'LaplacianKernel'); 
plot(xi,f);
hold on 
pd1 = makedist('Uniform'); %%Plot Uniform Distribution in red                    
pdf1 = pdf(pd1,xi);
plot(xi,pdf1,'r'); 
title('Laplacian Kernel h=0.1 b=0.1') 



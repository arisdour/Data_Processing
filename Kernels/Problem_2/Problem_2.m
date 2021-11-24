%%
clc;
clear;
close all; 

%%% Parameters 

h=0.01;
L=1;

%%% Load Stars and Circles from file
data1 = load('hw2-2data.mat');
data = [data1.circles;data1.stars];
stars=data(22:42,:);
circles=data(1:21,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Erotima c %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%% Calculate Kernel(xi,xj)
for i= 1:42
    for j=1:42
        a=[data(i ,1) data(i ,2)];
        b=[data(j ,1) data(j ,2)];
        d1(i,j)=(norm(a-b)).^2;
        Kernel(i,j)=exp(-(norm(b-a)).^2 / (h));
    end
end




y1=-ones(1,21);
y11=ones(1,21);
y= cat(1,y1',y11');

coeffs=inv((Kernel.' * Kernel + L* Kernel)) *Kernel*y ;

coeffs_b=coeffs(1:21);
coeffs_a=coeffs(22:42);

%%%Check classification reults 
for i= 1:21
    X=[stars(i,1),stars(i,2) ];
    class_stars(i,:) = phi(X ,h, coeffs_a,coeffs_b , stars, circles);
end


for i= 1:21
    X=[circles(i,1),circles(i,2) ];
    class_circles(i,:) = phi(X ,h, coeffs_a,coeffs_b , stars, circles);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Erotima E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=1;
step=0.01;
for i =-1.5:step:1.2
    for j=-1.5:step:2
        X=[i j];
        X1=[i+step j+step ];
        if phi(X ,h, coeffs_a,coeffs_b , stars, circles) *phi(X1 ,h, coeffs_a,coeffs_b , stars, circles)<0
            points(k,:)=(X+X1)/2;
            k=k+1;
        end
    end
end

figure()
x=points(:,1);
y=points(:,2);
plot(x,y,'.')
hold on 
data = load('hw2-2data.mat');
plot(data.circles(:,1).', data.circles(:,2).', 'o', data.stars(:,1).', data.stars(:,2).', '*');
title(['ö(÷)   h= ' num2str(h) ' ' 'L=' num2str(L)  '   Step:' num2str(step)]) 

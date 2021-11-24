function result = phi(X,h,coeffs_a , coeffs_b, Xstars, Xcircles)
    
    Stars=0;
    Circles=0;
    for i=1:21
        ai=coeffs_a(i);
        a=[Xstars(i ,1) Xstars(i ,2)];
        Kernel1=exp( - (( (X(1)-a(1)).^2) + ((X(2)-a(2)).^2))/(h));
        Stars= Stars+ai*Kernel1;
        bj=coeffs_b(i);
        b=[Xcircles(i ,1) Xcircles(i ,2)];        
        Kernel2=exp(- ( ((X(1)-b(1)).^2) + ((X(2)-b(2)).^2)) / (h));
        Circles=Circles+bj*Kernel2;
    end
    result=Stars+Circles;

    
end
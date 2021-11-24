function G = myGaussianKernel(x)
    h=0.1;
    G = (1/sqrt((2*pi*h))) * exp(-(x.^2)/(2*h));
end
function L = LaplacianKernel(x)
    h=0.1;
    L = (1/(2*h)) * exp(-abs(x)/(h));
end
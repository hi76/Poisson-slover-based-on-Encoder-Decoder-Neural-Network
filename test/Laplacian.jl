using Images

function _laplacian(a)
    kernel = [0 1 0; 1 -4 1; 0 1 0]
    grad2 = imfilter(a, kernel,Fill(0))
    return grad2
end

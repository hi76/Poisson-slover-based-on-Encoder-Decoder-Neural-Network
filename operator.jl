using Images

function _operator(a)
    kernel = [0 0 0 ;0 1 0 ;0 0 0; 0 1 0; 1 -6 1; 0 1 0; 0 0 0 ;0 1 0 ;0 0 0]
    kernel = reshape(kernel,3,3,3)
    grad2 = imfilter(a, kernel,Fill(0))
    return grad2
end

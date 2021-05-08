using Knet
using Images
const Atype = KnetArray{Float32}



include("operator.jl")
function operator3d_DataGenerate(data_nums,data_size;atype=Atype)
    center = rand(Float64, data_size-4, data_size-4, data_size-4, 1, data_nums)
    x = zeros(Float64, data_size, data_size, data_size, 1, data_nums)
    x[3:end-2,3:end-2,3:end-2,1,:] = center
    y = zeros(Float64, data_size, data_size, data_size, 1, data_nums)
    for i = 1:data_nums
        grad2 = _operator(x[:,:,:,1,i])
        y[:,:,:,1,i] = grad2
    end
    return convert.(atype,(y,x))
end

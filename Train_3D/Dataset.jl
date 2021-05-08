include("DataGenerate.jl")
include("make_minibatch.jl")
using Base.Iterators: repeated, partition

function Dataset(data_nums,data_size,batch_size)
    X,Y = operator3d_DataGenerate(data_nums,data_size)
    mb_idxs = collect(partition(1:data_nums, batch_size))
    x,y=[],[]
    for i in mb_idxs
        append!(x, [make_minibatch(X, Y, i)[1]])
        append!(y, [make_minibatch(X, Y, i)[2]])
    end
    return (x,y)
end

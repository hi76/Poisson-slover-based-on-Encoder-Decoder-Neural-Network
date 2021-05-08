function make_minibatch(X, Y, idxs)
    X_batch = KnetArray{Float32}(undef,size(X[:,:,:,:,1])..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:,:,i] = X[:,:,:,:,idxs[i]]
    end
    Y_batch = KnetArray{Float32}(undef,size(Y[:,:,:,:,1])..., length(idxs))
    for i in 1:length(idxs)
        Y_batch[:,:,:,:,i] = Y[:,:,:,:,idxs[i]]
    end
    return(X_batch, Y_batch)
end

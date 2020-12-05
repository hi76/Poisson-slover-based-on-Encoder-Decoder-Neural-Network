function _savemodel(model, count, date4)
    if count==0
        @info("saving model")
        Knet.save(date4*"trained_model.jld2", "model", model)
        @info("complete save "*date4*"trained_model.jld2")
    end
end

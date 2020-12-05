MSE_loss = []
using PyPlot
s = [j for j=1:10000]

function _showloss(loss, count, date)
    append!(MSE_loss, loss)
    if count==0
        @info("Ploting loss...")
        figure("MSE_loss")
        hist(MSE_loss,log=true)
        ylabel("Quantities",fontsize=15)
        grid(axis = "y")
        tick_params(labelsize=18)
        savefig(date*".png")
        close("MSE_loss")

        if length(MSE_loss)>2000
            figure("MSE_loss_little")
            ylabel("Quantities",fontsize=15)
            hist(MSE_loss[end-1500:end-1],log=true)
            grid(axis = "y")
            tick_params(labelsize=18)
            savefig(date*"_short.png")
            close("MSE_loss_little")
        end

        @info("complete plot loss")
    end
end

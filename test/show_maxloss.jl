max_loss = []
using PyPlot
s = [j for j=1:10000]

function _showmaxloss(maxloss, count, date)
    append!(max_loss, maxloss)
    if count==0
        @info("Ploting maxloss...")
        figure("max_loss")
        ylabel("Quantities",fontsize=15)
        hist(max_loss,log=true)
        grid(axis = "y")
        tick_params(labelsize=18)
        savefig(date*".png")
        close("max_loss")
        @info("complete plot maxloss")

        if length(max_loss)>2000
            figure("max_loss_little")
            ylabel("Quantities",fontsize=15)
            hist(max_loss[end-1500:end-1],log=true)
            grid(axis = "y")
            tick_params(labelsize=18)
            savefig(date*"_short.png")
            close("max_loss_little")
        end

    end
end

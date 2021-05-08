max_loss = []
using PyPlot

function _showmaxloss(maxloss, count, date)
    append!(max_loss, maxloss)
    @show maxloss
    if count==0
        @info("Ploting maxloss...")
        figure("max_loss")
        title("MAXloss")
        ylabel("loss")
        xlabel("train_nums")
        plot(max_loss)
        plt.grid(axis = "y")
        savefig(date*".png")
        close("max_loss")
        @info("complete plot maxloss")

        if length(max_loss)>2000
            figure("max_loss_little")
            title("MAXloss_1500")
            ylabel("loss")
            xlabel("train_nums")
            plot(max_loss[end-1500:end-1])
            plt.grid(axis = "y")
            savefig(date*"_short.png")
            close("max_loss_little")
        end

        if length(max_loss)>16000
            figure("max_loss_medium")
            title("MAXloss_15500")
            ylabel("loss")
            xlabel("train_nums")
            plot(max_loss[end-15500:end-1])
            plt.grid(axis = "y")
            savefig(date*"_long.png")
            close("max_loss_medium")
        end
        @info("complete plot maxloss")
    end
end

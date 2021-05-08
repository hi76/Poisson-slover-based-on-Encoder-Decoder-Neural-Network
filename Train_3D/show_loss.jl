MSE_loss = []
using PyPlot

function _showloss(mseloss, count, date)
    append!(MSE_loss, mseloss)
    @show mseloss
    if count==0
        @info("Ploting loss...")
        figure("MSE_loss")
        plot(MSE_loss)
        title("MSEloss")
        ylabel("loss")
        xlabel("train_nums")
        plt.grid(axis = "y")
        savefig(date*".png")
        close("MSE_loss")

        if length(MSE_loss)>2000
            figure("MSE_loss_little")
            title("MSEloss_1500")
            ylabel("loss")
            xlabel("train_nums")
            plot(MSE_loss[end-1500:end-1])
            plt.grid(axis = "y")
            savefig(date*"_short.png")
            close("MSE_loss_little")
        end

        if length(MSE_loss)>16000
            figure("MSE_loss_little")
            title("MSEloss_15500")
            ylabel("loss")
            xlabel("train_nums")
            plot(MSE_loss[end-15500:end-1])
            plt.grid(axis = "y")
            savefig(date*"_long.png")
            close("MSE_loss_little")
        end

        @info("complete plot mseloss")
    end
end

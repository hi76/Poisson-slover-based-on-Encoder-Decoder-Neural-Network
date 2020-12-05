include("Dataset.jl")
include("show_loss.jl")
include("show_maxloss.jl")
include("Plot_image.jl")
include("Model.jl")

function test_NN(path=".../",data_nums=10,image_size=16)
    batch_size=1

    date1 = path*"test/NN_MSELoss"
    date2 = path*"test/NN_MAXLoss"
    date3 = path*"test/NN_Result_"

    @info("load model...")
    model = Knet.load(path*"test/trained_model.jld2","model")
    @info("complete.")
    @info("generate data...")
    data_x,data_y = Dataset(data_nums,image_size,batch_size)
    @info("complete.")
    @info("start test...")
    for j = 1:length(data_x)
        x = data_x[j]
        y = data_y[j]
        mseloss = model(x,y)
        maxloss = model(x,y,abs)
        print(j,"   Mseloss:",mseloss,"   Maxloss:",maxloss, "\n")
        _showloss(mseloss,j%1,date1)
        _showmaxloss(maxloss,j%1,date2)
        cc=Int(data_nums/5)
        if j%cc==0
            _showimage(model(x), x, y, j, image_size, date3)
        end
    end

    @info("complete all.")
end

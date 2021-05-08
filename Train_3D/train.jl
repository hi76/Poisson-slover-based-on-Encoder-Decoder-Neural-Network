include("Model.jl")
include("Dataset.jl")
include("show_loss.jl")
include("show_maxloss.jl")
include("savemodel.jl")

function Train(path=".../",epochs=200,train_nums=1000,lr=0.00001,image_size=16,data_nums=20000,batch_size=10,read_data=false,read_model=false,show_nums=500)

    date1 = path*"Mseloss"
    date2 = path*"Maxloss"
    date3 = path*"Result"
    date4 = path

    if read_model==true
        @info("load model...")
        model = Knet.load(path*"trained3d_model.jld2", "model")
        @info("complete.")
    else
        @info("build new model...")
        model = Chain((Conv(3,1,32),IdentitySkip(3,32,Conv),Conv(3,32,64),IdentitySkip(3,64,Conv),Conv(3,64,128),IdentitySkip(3,128,Conv),Conv(3,128,256),IdentitySkip(3,256,Conv),IdentitySkip(3,256,deConv),deConv(3,256,128),IdentitySkip(3,128,deConv),deConv(3,128,64),IdentitySkip(3,64,deConv),deConv(3,64,32),IdentitySkip(3,32,deConv),deConv(3,32,1)))
        @info("complete.")
    end

    if read_data==true
        @info("load data...")
        data_x = Knet.load(path*"data_rho.jld2", "data_x")
        data_y = Knet.load(path*"data_phi.jld2", "data_y")
        @info("complete.")
    else
        @info("generate data...")
        data_x,data_y = Dataset(data_nums,image_size,batch_size)
        Knet.save(path*"data_rho.jld2", "data_x", data_x)
        Knet.save(path*"data_phi.jld2", "data_y", data_y)
        @info("complete.")
    end

    @info("Start Train...")
    len_data = length(data_x)
    for n = 1:train_nums
        for j = 1:len_data
            for i=1:epochs
                x = data_x[j]
                y = data_y[j]
                adam!(model,[(x,y)]; lr=lr)
                iter = (j-1)*epochs + i + (n-1)*len_data*epochs
                @show iter
                _showloss(model(x,y),((j-1)*epochs+i)%show_nums,date1)
                _showmaxloss(model(x,y,abs),((j-1)*epochs+i)%show_nums,date2)
                _savemodel(model, ((j-1)*epochs+i)%show_nums, date4)
            end
        end
    end

    @info("complete train.")

end

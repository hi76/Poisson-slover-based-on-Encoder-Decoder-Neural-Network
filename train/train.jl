include("Model.jl")
include("Dataset.jl")
include("show_loss.jl")
include("show_maxloss.jl")
include("Plot_image.jl")
include("savemodel.jl")

function Train(path=".../",epochs=50,train_nums=10,lr=0.001,image_size=16,data_nums=100000,batch_size=20,read_data=false,read_model=false,show_nums=500)

    date1 = path*"train/Mseloss"
    date2 = path*"train/Maxloss"
    date3 = path*"train/Result"
    date4 = path*"train/"

    if read_model==true
        @info("load model...")
        model = Knet.load(path*"trained_model.jld2", "model")
        @info("complete.")
    else
        @info("build new model...")
        model = Chain((Conv(3,1,28),IdentitySkip(3,28,Conv),Conv(3,28,64),IdentitySkip(3,64,Conv),Conv(3,64,128),IdentitySkip(3,128,Conv),Conv(3,128,256),IdentitySkip(3,256,Conv),IdentitySkip(3,256,deConv),deConv(3,256,128),IdentitySkip(3,128,deConv),deConv(3,128,64),IdentitySkip(3,64,deConv),deConv(3,64,28),IdentitySkip(3,28,deConv),deConv(3,28,1)))
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
                _showimage(model(reshape(x[:,:,:,1],(image_size,image_size,1,1))), x[:,:,:,1], y[:,:,:,1], ((j-1)*epochs+i)%show_nums, image_size, date3, "1")
                _showimage(model(reshape(x[:,:,:,2],(image_size,image_size,1,1))), x[:,:,:,2], y[:,:,:,2], ((j-1)*epochs+i)%show_nums, image_size, date3, "2")
                _showimage(model(reshape(x[:,:,:,3],(image_size,image_size,1,1))), x[:,:,:,3], y[:,:,:,3], ((j-1)*epochs+i)%show_nums, image_size, date3, "3")
                _showimage(model(reshape(x[:,:,:,4],(image_size,image_size,1,1))), x[:,:,:,4], y[:,:,:,4], ((j-1)*epochs+i)%show_nums, image_size, date3, "4")
                _showimage(model(reshape(x[:,:,:,5],(image_size,image_size,1,1))), x[:,:,:,5], y[:,:,:,5], ((j-1)*epochs+i)%show_nums, image_size, date3, "5")
                _showimage(model(reshape(x[:,:,:,6],(image_size,image_size,1,1))), x[:,:,:,6], y[:,:,:,6], ((j-1)*epochs+i)%show_nums, image_size, date3, "6")
                _showimage(model(reshape(x[:,:,:,7],(image_size,image_size,1,1))), x[:,:,:,7], y[:,:,:,7], ((j-1)*epochs+i)%show_nums, image_size, date3, "7")
                _showimage(model(reshape(x[:,:,:,8],(image_size,image_size,1,1))), x[:,:,:,8], y[:,:,:,8], ((j-1)*epochs+i)%show_nums, image_size, date3, "8")
                _showimage(model(reshape(x[:,:,:,9],(image_size,image_size,1,1))), x[:,:,:,9], y[:,:,:,9], ((j-1)*epochs+i)%show_nums, image_size, date3, "9")
                _showimage(model(reshape(x[:,:,:,10],(image_size,image_size,1,1))), x[:,:,:,10], y[:,:,:,10], ((j-1)*epochs+i)%show_nums, image_size, date3, "10")
                _showimage(model(reshape(x[:,:,:,11],(image_size,image_size,1,1))), x[:,:,:,11], y[:,:,:,11], ((j-1)*epochs+i)%show_nums, image_size, date3, "11")
                _showimage(model(reshape(x[:,:,:,12],(image_size,image_size,1,1))), x[:,:,:,12], y[:,:,:,12], ((j-1)*epochs+i)%show_nums, image_size, date3, "12")
                _showimage(model(reshape(x[:,:,:,13],(image_size,image_size,1,1))), x[:,:,:,13], y[:,:,:,13], ((j-1)*epochs+i)%show_nums, image_size, date3, "13")
                _savemodel(model, ((j-1)*epochs+i)%show_nums, date4)
            end
        end
    end

    @info("complete train.")

end

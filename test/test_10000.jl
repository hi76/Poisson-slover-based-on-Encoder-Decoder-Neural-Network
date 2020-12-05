include("Dataset.jl")
include("show_loss.jl")
include("show_maxloss.jl")
include("Plot_image.jl")
include("Model.jl")

function test_10000(path=".../",data_nums=10000,model_path)
    batch_size=1
    image_size=16
    ooo5=0
    oo1=0
    oo5=0
    model_path = path*"test/trained_model.jld2"

    date1 = path*"16_MSELoss"          #set the path of saving results
    date2 = path*"16_MAXLoss"
    date3 = path*"Situation_"

    @info("load model...")          #load model
    model = Knet.load(model_path,"model")
    @info("complete.")
    @info("generate data...")      #generate data
    data_x,data_y = Dataset_7(data_nums,image_size,batch_size)
    @info("complete.")
    @info("start test...")
    for j = 1:length(data_x)
        x = data_x[j]
        y = data_y[j]
        mseloss = model(x,y)
        maxloss = model(x,y,abs)
        print(j,"   Mseloss:",mseloss,"   Maxloss:",maxloss, "\n")
        if maxloss>0.0005
            ooo5+=1
        end
        if maxloss>0.001
            oo1+=1
        end
        if maxloss>0.005
            oo5+=1
        end
        print(">.ooo5: ",ooo5,"    >.oo1: ",oo1,"    >.oo5: ",oo5,"\n")     #show nums for different precision
        _showloss(mseloss,j%2000,date1)          #draw MSEloss
        _showmaxloss(maxloss,j%2000,date2)       #draw MAXloss
        if j == data_nums                        #draw result examples
            _showimage(model(data_x[j-1]), data_x[j-1], data_y[j-1], j, image_size, date3*"1")
            _showimage(model(data_x[j-2]), data_x[j-2], data_y[j-2], j, image_size, date3*"2")
            _showimage(model(data_x[j-3]), data_x[j-3], data_y[j-3], j, image_size, date3*"3")
            _showimage(model(data_x[j-4]), data_x[j-4], data_y[j-4], j, image_size, date3*"4")
            _showimage(model(data_x[j-5]), data_x[j-5], data_y[j-5], j, image_size, date3*"5")
            _showimage(model(data_x[j-6]), data_x[j-6], data_y[j-6], j, image_size, date3*"6")
            _showimage(model(data_x[j-7]), data_x[j-7], data_y[j-7], j, image_size, date3*"7")
            _showimage(model(data_x[j-8]), data_x[j-8], data_y[j-8], j, image_size, date3*"8")
            _showimage(model(data_x[j-9]), data_x[j-9], data_y[j-9], j, image_size, date3*"9")
            _showimage(model(data_x[j-10]), data_x[j-10], data_y[j-10], j, image_size, date3*"10")
            _showimage(model(data_x[j-11]), data_x[j-11], data_y[j-11], j, image_size, date3*"11")
            _showimage(model(data_x[j-12]), data_x[j-12], data_y[j-12], j, image_size, date3*"12")
            _showimage(model(data_x[j-13]), data_x[j-13], data_y[j-13], j, image_size, date3*"13")
        end
    end

    @info("all complete.")


end

include("Dataset.jl")
include("show_loss.jl")
include("show_maxloss.jl")
include("Plot_image.jl")
include("Model.jl")



function test_iso(path=".../")
    data_nums = 1
    image_size=16
    batch_size = 1

    date = path*"test/iso_"

    @info("load model...")
    model = Knet.load(path*"test/trained_model.jld2","model")
    @info("complete.")
    @info("generate data...")
    fakenorm_rho,norm_phi,true_rho,true_phi,delta_true_phi,max_true_phi = Dataset_iso(data_nums,image_size,batch_size)
    @info("complete.")
    @info("start test...")
    j=1
    x = fakenorm_rho[j]
    y = norm_phi[j]
    mseloss = model(x,y)
    maxloss = model(x,y,abs)
    print(j,"   Mseloss:",mseloss,"   Maxloss:",maxloss, "\n")

    _showimage(model(x), x, y, j, image_size, date*"_model_")

    figure("true_rho")
    imshow(true_rho)
    xticks([])
    yticks([])
    cb=colorbar()
    cb.ax.tick_params(labelsize=26)
    savefig(date*"_true_rho.png")
    close("true_rho")
    figure("true_phi")
    imshow(true_phi)
    xticks([])
    yticks([])
    cb=colorbar()
    cb.ax.tick_params(labelsize=26)
    savefig(date*"_true_phi.png")
    close("true_phi")
    predict_norm_phi = model(x)
    predict_norm_phi = Array(predict_norm_phi)
    predict_norm_phi = reshape(predict_norm_phi,(size(predict_norm_phi)[1],size(predict_norm_phi)[2]))
    figure("predictback_truephi_difference")
    imshow(((predict_norm_phi.*delta_true_phi).+max_true_phi)[3:end-2,3:end-2]-true_phi)
    xticks([])
    yticks([])
    cb=colorbar()
    cb.ax.tick_params(labelsize=26)
    cb.formatter.set_scientific(true)
    cb.formatter.set_powerlimits((-2,0))
    cb.ax.yaxis.get_offset_text().set_fontsize(26)
    savefig(date*"error.png")
    close("predictback_truephi_difference")
    figure("predictback_truephi")
    imshow(((predict_norm_phi.*delta_true_phi).+max_true_phi)[3:end-2,3:end-2])
    xticks([])
    yticks([])
    cb=colorbar()
    cb.ax.tick_params(labelsize=26)
    savefig(date*"_predict_phi.png")
    close("predictback_truephi")


    @info("complete all.")
end

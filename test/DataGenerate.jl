using Knet
using Images
using SparseArrays,LinearAlgebra,Statistics
const Atype = KnetArray{Float32}

function make_data(data_nums,data_size,sorts)

    if (sorts==0)||(sorts==4)||(sorts==6)||(sorts==10)
        data = rand(Float64, data_size, data_size, 1, data_nums)
    end

    if sorts == 1
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[5:end-4,5:end-4,1,:] .= 0.0
    end

    if sorts == 3
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[1:end-10,11:end,1,:] .= 0.0
        data[11:end,1:end-10,1,:] .= 0.0
        data[1:end-10,1:end-10,1,:] .= 0.0
        data[11:end,11:end,1,:] .= 0.0
    end

    if sorts == 5
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[5:end-4,5:end-4,1,:] .= 0.1
    end

    if sorts == 7
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[1:end-10,11:end,1,:] .= 0.1
        data[11:end,1:end-10,1,:] .= 0.1
        data[1:end-10,1:end-10,1,:] .= 0.1
        data[11:end,11:end,1,:] .= 0.1
    end

    if sorts == 9
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[5:end-4,5:end-4,1,:] .= 0.99
    end

    if sorts == 11
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[1:end-10,11:end,1,:] .= 0.98
        data[11:end,1:end-10,1,:] .= 0.98
        data[1:end-10,1:end-10,1,:] .= 0.98
        data[11:end,11:end,1,:] .= 0.98
    end

    if (sorts==12)||(sorts==2)||(sorts==8)
        data = rand(Float64, data_size, data_size, 1, data_nums)
        data[4:end-3,4:end-3] = imfilter(data[4:end-3,4:end-3],Kernel.gaussian(2))
    end

    return data
end

include("Laplacian.jl")
function laplacian_DataGenerate_7(data_nums,data_size;atype=Atype)
    x = zeros(Float64, data_size, data_size, 1, data_nums)
    center = zeros(Float64,data_size-4,data_size-4,1,1)
    for i = 1:data_nums
        center = make_data(1,data_size-4,i%13)
        x[3:end-2,3:end-2,1,i] = center
    end
    y = zeros(Float64, data_size, data_size, 1, data_nums)
    for i = 1:data_nums
        grad2 = _laplacian(x[:,:,1,i])
        y[:,:,1,i] = grad2
    end
    return convert.(atype,(y,x))
end


function laplacian_DataGenerate(data_nums,data_size;atype=Atype)
    center = rand(Float64, data_size-4, data_size-4, 1, data_nums)
    x = zeros(Float64, data_size, data_size, 1, data_nums)
    x[3:end-2,3:end-2,1,:] = center
    y = zeros(Float64, data_size, data_size, 1, data_nums)
    for i = 1:data_nums
        grad2 = _laplacian(x[:,:,1,i])
        y[:,:,1,i] = grad2
    end
    return convert.(atype,(y,x))
end

function laplacian_DataGenerate_iso(data_nums,data_size;atype=Atype)

    u=1/((data_size-4)/2)
    G=2.8888888888888
    DT0=Float32(u)
    p=Float32(pi)
    function isochrone_potential(;M=1,b=1) #Cylindrical coordinates
        phi(r)=-G*M/(b+sqrt(b^2+r^2))

        function rho(r)
            a=sqrt(b^2+r^2)
            return M*(2*(b+a)*a^2-r^2*(b+3a))/(4*p*(b+a)^3*a^3)
        end

        x=[j for i=-1:DT0:1-DT0, j=-1:DT0:1-DT0];
        y=[i for i=-1:DT0:1-DT0, j=-1:DT0:1-DT0];
        r=sqrt.(x.^2+y.^2);
        im_rho=rho.(r);
        im_phi=phi.(r);
        min_im_phi = minimum(im_phi)
        max_im_phi = maximum(im_phi)
        delta_im_phi =  min_im_phi - max_im_phi
        norm_im_phi = (im_phi .- max_im_phi) ./ delta_im_phi
        return im_rho,im_phi,norm_im_phi,delta_im_phi,max_im_phi;
    end


    true_rho,true_phi,norm_phi,delta_true_phi,max_true_phi = isochrone_potential(;M=1,b=1)
    norm_phi = reshape(norm_phi,(size(norm_phi)[1],size(norm_phi)[2],1,1))
    x = zeros(Float64, data_size, data_size, 1, data_nums)
    x[3:end-2,3:end-2,1,:] = norm_phi
    y = zeros(Float64, data_size, data_size, 1, data_nums)
    for i = 1:data_nums
        grad2 = _laplacian(x[:,:,1,i])
        y[:,:,1,i] = grad2
    end
    return convert.(atype,(y,x)),true_rho,true_phi,delta_true_phi,max_true_phi
end

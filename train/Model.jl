using Knet
using Statistics
struct Conv
    w
    b
    a
end
(c::Conv)(x) = c.a .* relu.(conv4(c.w, x,padding=1)) .+ c.b
Conv(kernel_size::Int,in_channel::Int,out_channel::Int) = Conv(param(kernel_size,kernel_size,in_channel,out_channel;atype=Atype), param0(1,1,out_channel,1;atype=Atype), param(1,1,out_channel,1;atype=Atype))

struct deConv
    w
    b
    a
end
(d::deConv)(x) = d.a .* relu.(deconv4(d.w, x,padding=1)) .+ d.b
deConv(kernel_size::Int,in_channel::Int,out_channel::Int) = deConv(param(kernel_size,kernel_size,out_channel,in_channel;atype=Atype), param0(1,1,out_channel,1;atype=Atype), param(1,1,out_channel,1;atype=Atype))

struct pad0Conv
    w
    b
end
(c::pad0Conv)(x) = relu.(conv4(c.w, x,padding=0) .+ c.b)
pad0Conv(kernel_size,in_channel,out_channel) = pad0Conv(param(kernel_size,kernel_size,in_channel,out_channel;atype=Atype), param(1,1,out_channel,1;atype=Atype))


struct Chain
    layers
end
(c::Chain)(x) = (for l in c.layers;x = l(x); end; x)
(c::Chain)(x,y) = mean(abs2.(c(x).-y))
(c::Chain)(x,y,z) = maximum(z.(c(x).-y))


struct IdentitySkip
    inner
end
(m::IdentitySkip)(x) = m.inner(x) .+ x
IdentitySkip(kernel_size,keep_channel,f) = IdentitySkip(f(kernel_size,keep_channel,keep_channel))

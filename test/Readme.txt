#trained_model.jld2 is a well-trained Encoder-Decoder model.
#It's trained on 16*16 size data

#To test model
#First , enter in your julia. then type these below. Being care of your path.





#For testing 10 thousand data.
#Show test result: MSELoss,MAXLoss for all data and 13 result examples.
##############################usage################################In Julia

include("Your path/test/test_10000.jl")
test_10000("Your path/")
#or
test_10000("Your path/",data_nums)
#data_nums is your choice

###########################Once you see "complete all.", test over###############






#For testing different size data.
##############################usage################################In Julia

include("Your path/test/test_NN.jl")
test_NN("Your path/")
#or
test_NN("Your path/",data_nums,image_size)
#data_nums and image_size are depend on you.
#you can try image_size (32/64/128/256/512/1025)

###########################Once you see "complete all.", test over###############






#For testing isochrone_potential problem
##############################usage################################In Julia

include("Your path/test/test_iso.jl")
test_iso("Your path/")

###########################Once you see "complete all.", test over###############






###########################Package installation##############################
]
add PyPlot,Knet,Statistics,Images
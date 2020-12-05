# Poisson-slover-based-on-Encoder-Decoder-Neural-Network

#trained_model.jld2 is a well-trained Encoder-Decoder model.
#It's trained on 16*16 size data

#To test model
#First , enter in your julia. then type these below. Being care of your path.
#For testing 10 thousand data.
#Show test result: MSELoss,MAXLoss for all data and 13 result examples.
#usage In Julia

include("Your path/test/test_10000.jl")
test_10000("Your path/")
#or
test_10000("Your path/",data_nums)

#data_nums is your choice
#Once you see "complete all.", test over

#For testing different size data.
#usage

include("Your path/test/test_NN.jl")
test_NN("Your path/")
#or
test_NN("Your path/",data_nums,image_size)

#data_nums and image_size are depend on you.
#you can try image_size (32/64/128/256/512/1025)
#Once you see "complete all.", test over

#For testing isochrone_potential problem

include("Your path/test/test_iso.jl")
test_iso("Your path/")

#Once you see "complete all.", test over

#Package installation
]
add PyPlot,Knet,Statistics,Images

#Train a Encoder-Decoder Neural Network model.
#usage
#parameters:

path = "/Your path/"               #don't forget the first and the last '/'.
epochs = a number                #Default 50, it means that each data train 50 times, then step to next data.
train_nums = a number          #Default 10, it means that train process will iterate 10 times.
lr = a number                         #Default 0.001, it is the learning rate.
image_size = a number           #Default 16, it means that model will trained on 16*16 size data.
data_nums = a number           #Default 100000, it is the amount of training data.
batch_size = a number            #Default 20,it is the batch size.
read_data = true/false            #Default false, it will generate new data each running, if set to true, it will train with data in the last train.
read_model = true/false         #Default false, if set to true, program will continue train model based on your last train.
show_nums = a number         #Default 500, it means that mseloss.png maxloss.png result.png are drew every 500 train steps. Meantime the trained model will be saved as trained_model.jld2.

#Once you see"Complete train.", train over

#If you want to test your trained model, just copy trained_model.jld2 to test file and run test_**.jl.

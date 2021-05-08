#Here is the model for 3D data training.

#the usage is seem to train

###############################usage###############################################

include("train.jl")
Train(***)

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


####################Once you see"Complete train.", train over########################

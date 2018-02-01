###################################################################################
######################## 1. load the libraries ####################################
###################################################################################
library(mxnet)
library(imager)
#library(EBImage)
library(pbapply)
library(caret)

###################################################################################
######################## 2. Data Preprocessing:                           #########
############################# Download the training dataset from          #########
############################# https://www.kaggle.com/c/nos-vs-fps/data  #########
###################################################################################
## Directory for images
image_dir <- "train"

## Set width and height for resizing images
width <- 32
height <- 32

## extract_feature
extract_feature <- function(dir_path, width, height, is_fp = TRUE, add_label = TRUE) {
  img_size <- width*height
  # List images in path
  images_names <- list.files(dir_path)
  if (add_label) {
    # Select only fps or nos images
    images_names <- images_names[grepl(ifelse(is_fp, "fp", "no"), images_names)]
    # Set label, fp = 0, no = 1
    label <- ifelse(is_fp, 0, 1)
  }
  print(paste("Start processing", length(images_names), "images"))
  # This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    # Read image
    #img <- readImage(file.path(dir_path, imgname))
    img <- load.image(file.path(dir_path, imgname))
    # Resize image
    #img_resized <- EBImage::resize(img, w = width, h = height)
    img_resized <- imager::resize(img,size_x = width,size_y = height)
    # Set to grayscale
    #grayimg <- channel(img_resized, "gray")
    if (length(channels(img_resized)) == 1)
    {
      grayimg <- img_resized
    }else if(length(channels(img_resized)) > 1 && length(channels(img_resized)) < 4)
    {
      grayimg <- grayscale(img_resized)
    }else{
      grayimg <- grayscale(channel(img_resized,1:3))
    }
    # Get the image as a matrix
    #img_matrix <- grayimg@.Data
    img_matrix <- as.matrix(grayimg)
    # Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  # bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  # Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if (add_label) {
    # Add label
    feature_matrix <- cbind(label = label, feature_matrix)
  }
  return(feature_matrix)
}

# Process fp and no images separately and save them into data.frame
fps_data <- extract_feature(dir_path = image_dir, width = width, height = height)
nos_data <- extract_feature(dir_path = image_dir, width = width, height = height, is_fp = FALSE)

# Save the data just in case
saveRDS(fps_data, "model/fp.rds")
saveRDS(nos_data, "model/no.rds")

#####################################################################
################## 3. Model Training: Data partitions:  ############
####################### randomly split 90% of data into  ############
####################### training set with equal weights  ############
####################### for fps and nos, and the rest  ############
####################### 10% will be used as the test set.############
#####################################################################
## Bind rows in a single dataset
complete_set <- rbind(fps_data, nos_data)

## test/training partitions
training_index <- createDataPartition(complete_set$label, p = .9, times = 1)
training_index <- unlist(training_index)
train_set <- complete_set[training_index,]
test_set <- complete_set[-training_index,]

## Reshape the data into a proper format required by the model:
train_data <- data.matrix(train_set)
train_x <- t(train_data[, -1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(height, width, 1, ncol(train_x))

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(height, width, 1, ncol(test_x))

## Training the model:
# Model
mx_data <- mx.symbol.Variable('data')
# 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data = mx_data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2,2 ))
# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 100)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)
# Set seed for reproducibility
mx.set.seed(100)
# Device used. Sadly not the GPU :-(
device <- mx.cpu()
# Train on 1200 samples
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 160,
                                     array.batch.size = 25,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
#                                     epoch.end.callback = mx.callback.save.checkpoint("model/mxnetfp"))
                                     epoch.end.callback = mx.callback.log.train.metric(100))
# After 30 iterations, this model achieves a peak performance of about 85% accuracy. 

## Test test set
predict_probs <- predict(model, test_array)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test_data[, 1], predicted_labels)
sum(diag(table(test_data[, 1], predicted_labels)))/2500
# The model reaches 75% accuracy on the test set. 

###############################################################
################### 4. Application ############################
###############################################################

feature_image <- function(img_name) {
  # Read image
  #img <- readImage(img_path)
  img <- load.image(img_name)
  # Resize image
  #img_resized <- resize(img, w = width, h = height)
  img_resized <- imager::resize(img, size_x = width, size_y = height)
  # Set to grayscale
  #grayimg <- channel(img_resized, "gray")
  if (length(channels(img_resized)) == 1)
  {
    grayimg <- img_resized
  }else if(length(channels(img_resized)) > 1 && length(channels(img_resized)) < 4)
  {
    grayimg <- grayscale(img_resized)
  }else{
    grayimg <- grayscale(channel(img_resized,1:3))
  }
  # Get the image as a matrix
  #img_matrix <- grayimg@.Data
  img_matrix <- as.matrix(grayimg)
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  return(img_vector)
}

images_names <- list.files("test",recursive = TRUE)

for (i in 1:length(images_names))
{
  iferror <- tryCatch({
    img_vector <- feature_image(file.path("test/",images_names[i]))},
    error = function(e) {return("yes")})
  if (class(iferror) == "character") {
    if (iferror == "yes") next;
  }
  
  img_matrix <- t(data.matrix(img_vector))
  dim(img_matrix) <- c(height, width, 1, 1)
  # load the model
  #model <- mx.model.load("mxnetfp", iteration=30)
  # prediction
  flag <- max.col(t(predict(model, img_matrix))) - 1
  Cfile <- unlist(strsplit(images_names[i],split="[/]"))
  if (flag == 0)
  {
    file.copy(file.path("test",images_names[i]),paste0("result/fps/",Cfile[length(Cfile)]),overwrite = TRUE)
  }else{
    file.copy(file.path("test",images_names[i]),paste0("result/nos/",Cfile[length(Cfile)]),overwrite = TRUE)
  }
  cat(i,flag,"\n")
}

# only used for testing
images_names <- list.files("test",recursive = TRUE)
for (i in 1:length(images_names))
{
  img_vector <- feature_image(file.path("test/",images_names[i]))
  img_matrix <- t(data.matrix(img_vector))
  dim(img_matrix) <- c(height, width, 1, 1)
  flag <- max.col(t(predict(model, img_matrix))) - 1
  cat(images_names[i],flag,"\n")
}

#used for im object testing
img2 <- load.image("train/fp4.jpg") %>% grayscale()
img <- (img2<0.2 & img2>0.1)
img <- 1-img
img_resized <- imager::resize(img, size_x = width, size_y = height)
if (length(channels(img_resized)) == 1)
{
  grayimg <- img_resized
}else if(length(channels(img_resized)) > 1 && length(channels(img_resized)) < 4)
{
  grayimg <- grayscale(img_resized)
}else{
  grayimg <- grayscale(channel(img_resized,1:3))
}
# Get the image as a matrix
#img_matrix <- grayimg@.Data
img_matrix <- as.matrix(grayimg)
# Coerce to a vector
img_vector <- as.vector(t(img_matrix))
img_matrix <- t(data.matrix(img_vector))
dim(img_matrix) <- c(height, width, 1, 1)
flag <- max.col(t(predict(model, img_matrix))) - 1
cat(flag,predict(model, img_matrix)[1])
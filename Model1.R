#Model 1. Copy of de Imdb sentiment example: https://keras.rstudio.com/articles/examples/imdb_lstm.html
#
#author: Kees van Eijden
#version/date:  0.2/18-feb-2018

library(stringr)
library(dplyr)
library(keras)
library(caret)


#reading traindata schoot-lgmm-ptsd
cat('Loading data..schoot-lgmm-ptsd.\n')
train_data      <- read.csv(file = "./data/example_dataset_1/csv/schoot-lgmm-ptsd-traindata.csv",
                            header= TRUE, sep= ",", stringsAsFactors = FALSE)
train_data      <- train_data %>% select(title, abstract, included_ats, included_final)

train_data$text <- paste0(train_data$abstract, train_data$title)
train_data      <- train_data %>% select(text, included_ats, included_final)



##contrary to the idm example, we don't have a vocabulary yet
##we use the keras tokenizer to create a vocabulary from the words in the abstracts and the titles
##every word is assigned a integer token
##the abstracts and titles are are transformed into sequences (vectors of tokens)
library(keras)
max_features <- 10000 
tokenizer    <- text_tokenizer(num_words = max_features)
tokenizer    %>% fit_text_tokenizer(train_data$text)

##tokenizer$word_index gives mapping of words to their tokens (=rank)
##there are many stop words 
#make ratio of included/not included more equal


sequences<- texts_to_sequences(tokenizer, train_data$text)

#we need sequences with fixed length. 
#the 75% quantile of lengths of all tokenized text is used. Shorter sequences are padded with 0
#this will favors titles because they are put in front of abstracts in the text strings
cat('Pad sequences................\n')
q75_text_len  <- quantile(sapply(sequences, length))[4]

sequences     <- pad_sequences(sequences, maxlen= q75_text_len, padding = "post", truncating = "post")



#make train en test sets
cat("Make train en test set with ratio 9:1 .......\n")
no_samples    <- dim(sequences)[1]
test_size     <- as.integer(no_samples * 0.10)
TESTSET       <- sample(1:no_samples, test_size)

x_train       <- sequences[-TESTSET, ]
y_train_ats   <- as.numeric(train_data$included_ats[-TESTSET])
y_train_final <- as.numeric(train_data$included_final[-TESTSET])

x_test        <- sequences[TESTSET, ]
y_test_ats    <- as.numeric(train_data$included_ats[TESTSET])
y_test_final  <- as.numeric(train_data$included_final[TESTSET])

cat("Overview of train en test tensors ......................\n")
cat('x_train tensor: ', length(dim(x_train)), ' shape:', dim(x_train), '\n')
cat('x_test  tensor: ', length(dim(x_test)), ' shape:', dim(x_test), '\n')


cat("Building model ........\n")
model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = max_features, output_dim = 64) %>% 
    layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
    loss      = 'binary_crossentropy',
    optimizer = 'adam',
    metrics   = c('accuracy')
)

cat("Train the model on abstracts as features and included after title screening as labels .......\n")
model %>% fit(
    x_train, y_train_ats,
    batch_size       = 128,
    epochs           = 15,
    validation_split = 0.2 
    #validation_data = list(x_test, y_test)
)

cat("Evaluate model on test set: ........\n")
scores <- model %>% evaluate(
    x_test, y_test_ats,
    batch_size = 128
)

cat('Test loss on test set:', scores[[1]])
cat('Test accuracy on test set', scores[[2]])

prediction <- predict_classes(model, x_test)

p          <- factor(x=prediction, levels= c("1", "0"))
r          <- factor(x=y_test_ats, levels= c("1", "0"))
conf_mod   <- confusionMatrix(p, r, mode= "everything", dnn= c("Prediction", "ATS"))
cat('Confusion matrix, prediction on data in test set (x_test) compared to labels (included_ats) of test set (y_test_ats):\n')
conf_mod

cat("End of model 1 .................\n")


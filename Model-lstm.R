#Model 1. Copy of de Imdb sentiment example: https://keras.rstudio.com/articles/examples/imdb_lstm.html
#
#author: Kees van Eijden
#version/date:  0.1/05-feb-2018

library(stringr)
library(dplyr)
library(keras)

#reading traindata schoot-lgmm-ptsd
cat('Loading data..schoot-lgmm-ptsd.\n')
train_data_raw  <- read.csv(file = "./data/example_dataset_1/csv/schoot-lgmm-ptsd-traindata.csv",
                        header= TRUE, sep= ",", stringsAsFactors = FALSE)
train_data      <- select(train_data_raw, abstract, title, included_ats, included_final)


##contrary to the idm example, we don't have a vocabulary yet
##we use the keras tokenizer to create a vocabulary from the words in the abstracts and the titles
##every word is assigned a integer token
##the abstracts and titles are are transformed into sequences (vectors of tokens)
library(keras)
max_features <- 20000 
tokenizer <- text_tokenizer(num_words = max_features)
title_and_abstract  <- paste0(train_data$title, train_data$abstract) #build vocabulary on words in titles and abstracts
tokenizer %>% fit_text_tokenizer(title_and_abstract)

##tokenizer$word_index gives mapping of words to their tokens (=rank)
##there are many stop words 


sequences_a <- texts_to_sequences(tokenizer, train_data$abstract)
sequences_t <- texts_to_sequences(tokenizer, train_data$title)

#we need sequences with fixed length. 
#the 75% quantile of lengths of all tokenized abstracts (titles ) is used. Shorter sequences are padded with 0
cat('Pad sequences................\n')
q75_title_len       <- quantile(sapply(sequences_t, length))[4]
q75_abstract_len    <- quantile(sapply(sequences_a, length))[4]

sequences_a <- pad_sequences(sequences_a, maxlen= q75_abstract_len, padding = "post", truncating = "post")
sequences_t <- pad_sequences(sequences_t, maxlen= q75_title_len, padding = "post", truncating = "post")

#make train en test sets
cat("Make train en test set with ratio 9:1 .......\n")
no_samples  <- dim(sequences_a)[1]
test_size   <- as.integer(no_samples * 0.10)
TESTSET     <- sample(1:no_samples, test_size)

x_train_a <- sequences_a[-TESTSET, ]
x_train_t <- sequences_t[-TESTSET, ]
y_train_ats   <- as.numeric(train_data$included_ats[-TESTSET])
y_train_final <- as.numeric(train_data$included_final[-TESTSET])

x_test_a <- sequences_a[TESTSET, ]
x_test_t <- sequences_t[TESTSET, ]
y_test_ats   <- as.numeric(train_data$included_ats[TESTSET])
y_test_final <- as.numeric(train_data$included_final[TESTSET])

cat("Overview of train en test tensor ......................\n")
cat('x_train_a tensor: ', length(dim(x_train_a)), ' shape:', dim(x_train_a), '\n')
cat('x_train_t tensor: ', length(dim(x_train_t)), ' shape:', dim(x_train_t), '\n')
cat('x_test_a tensor: ', length(dim(x_test_a)), ' shape:', dim(x_test_a), '\n')
cat('x_test_t tensor: ', length(dim(x_test_t)), ' shape:', dim(x_test_t), '\n')


cat("Building model ........\n")
model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = max_features, output_dim = 128) %>% 
    layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
)

batch_size <- 32 #is this the coirrect size for our case?

cat("Train the model on abstracts as features and included after title screening as labels .......\n")
model %>% fit(
    x_train_a, y_train_ats,
    batch_size = batch_size,
    epochs = 15,
    validation_split = 0.2  #guess
    #validation_data = list(x_test, y_test)
)

cat("Evaluate model on test set: ........\n")
scores <- model %>% evaluate(
    x_test_a, y_test_ats,
    batch_size = batch_size
)

cat('Test loss on test set:', scores[[1]])
cat('Test accuracy on test set', scores[[2]])

cat("End of model 1 .................\n")


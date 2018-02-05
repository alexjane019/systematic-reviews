#Systematic Review with LSTM. Copy of de Imdb sentiment example
#author: Kees van Eijden
#version/date:  0.1/04-feb-2018

#input files (.json) are assumed to be in the working directory


library(rjson)
r3 <- fromJSON(file = "./03. finally included.json")
r2 <- fromJSON(file = "./02. included after title screening.json")
r1 <- fromJSON(file = "./01. Initial Search.json")

#construct a vector of titles of positive examples after title screening
positives <- c()
for (i in 1:length(r2$title)) {
    positives <- append(positives, r2[["title"]][[i]])
}

#find the corresponding articles in the initial search set and set sentiment to 1
for (i in 1:length(r1$title)) {
    if (toString(r1$title[i]) %in% positives) {
        r1$sentiment[i] <- 1
    } else { r1$sentiment[i] <- 0}
}

##create two vectors: abstracts and corresponding sentiments
##delete samples with missing abstracts
abstracts <- c()
sentiments <- c()
for (i in 1:length(r1$abstract)) {
    if( !is.null(r1$abstract[[i]])) {
     abstracts <- append(abstracts, r1$abstract[[i]])
     sentiments <- append(sentiments, r1$sentiment[[i]])
    }
}

##the number of positive abstracts is very small
##compared to the number of negatives.
##this sets a very high treshold for the accuracy of the final model
t <- table(sentiments)
pos_frac <- t["1"]/(t["0"]+t["1"])
cat("Fraction positives: ", pos_frac)

##we don't have a vocabulary yet
##we use the keras tokenizer to create a vocabulary
##and corresponding (integer) tokens
##the abstracts are transformed into sequences (vectors of tokens)
library(keras)
num_words <- 5000 ### number of most important words, 5000 is just a guess!
tokenizer <- text_tokenizer(num_words = num_words)
tokenizer %>% fit_text_tokenizer(abstracts)
sequences <- texts_to_sequences(tokenizer, abstracts)

##tokenizer$word_index givess mapping of words to their index(=rank)
##there are many stop words

#make train en test sets
test_size <- 100 ###just a guess
TESTSET <- sample(1:length(sequences), test_size)

x_train <- sequences[-TESTSET]
y_train <- sentiments[-TESTSET]
x_test <- sequences[ TESTSET]
y_test <- sentiments[ TESTSET]



y_train <- as.numeric(y_train)
y_test  <- as.numeric(y_test)

max_len <- 80
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')


#building the LSTM model. Just a copy of the IMDB example
model <- keras_model_sequential()
model %>%
    layer_embedding(input_dim = num_words, output_dim = 128) %>% 
    layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
)

batch_size <- 32

model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = 15,
    validation_split = 0.2
    #validation_data = list(x_test, y_test)
)

scores <- model %>% evaluate(
    x_test, y_test,
    batch_size = batch_size
)

cat('Test loss:', scores[[1]])
cat('Test accuracy', scores[[2]])



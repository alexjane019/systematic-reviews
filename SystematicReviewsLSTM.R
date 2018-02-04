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

##tokenizer$word_index givess mapping of words to there index(=rank)
##there are many stopwords

#make train en test sets
test_size <- 100 ###just a guess
TESTSET <- sample(1:length(sequences), test_size)

train_sequences  <- sequences[-TESTSET]
train_sentiments <- sentiments[-TESTSET]
test_sequences   <- sequences[ TESTSET]
test_sentiments  <- sentiments[ TESTSET]

##vectorize to one-hot encode and convert samples into vectors
##may be keras provides a function to do this but 1 haven't found it yet

vectorize_sequences <- function(sequences, dimension = num_words) {
    # Creates an all-zero matrix of shape (length(sequences), dimension)
    results <- matrix(0, nrow = length(sequences), ncol = dimension) 
    for (i in 1:length(sequences))
        # Sets specific indices of results[i] to 1s
        results[i, sequences[[i]]] <- 1 
    results
}

train_samples <- vectorize_sequences(train_sequences, num_words)
test_samples  <- vectorize_sequences( test_sequences, num_words)
train_sentiments <- as.numeric(train_sentiments)
test_sentiments  <- as.numeric( test_sentiments)

#building the model. Just a copy of the IMDB example
model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = c(num_words)) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
)
#train the model
history <- model %>% fit(
    train_samples,
    train_sentiments,
    epochs = 20,             #values from IMDB example
    batch_size = 512,
    validation_split = 0.2
    #validation_data = list(x_val, y_val) doesn't work in latest keras package (2.1.3)
)

#how does the model perfom on the test set?
results <- model %>% evaluate(test_samples, test_sentiments)
results

##The accuracy on test set is 95%. That is very bad. A lot have to be done!

#
# ## dense matrix operations
# dummies <- dummyVars(~ ., data = mtrain)
# mtrain = predict(dummies, newdata = mtrain)
# mtest = predict(dummies, newdata = mtest)
#

## Inefficent - needs work
# d.event.wide <- dcast.data.table(data = d.event,
#                               id ~ event_type,
#                               value.var = "event_type")

### Using sparse.model.matrix - still needs work
## model.matrix(~ -1 + . , data=yourdata)
# d.event.wide <- sparse.model.matrix(id ~ event_type - 1, data = d.event)
#
# aggregate.sparse.matrix <- function(df, )

library(magrittr)
library(data.table)
library(caret)
library(Matrix)
library(dplyr)

time.start <- Sys.time()

logfile <- "./output/xgblog.txt"

if (!file.exists(logfile)) {
    file.create(logfile)
}

map <- plyr::mapvalues

d.event <- read.csv("input/event_type.csv") %>% data.table() %>% arrange(id)
d.feature <- read.csv("input/log_feature.csv") %>% data.table() %>% arrange(id)
d.resource <- read.csv("input/resource_type.csv") %>% data.table() %>% arrange(id)
d.severity <- read.csv("input/severity_type.csv") %>% data.table() %>% arrange(id)
d.test <- read.csv("input/test.csv") %>% data.table() %>% arrange(id)
d.train <- read.csv("input/train.csv") %>% data.table() %>% arrange(id)
d.set <- rbind(d.train, mutate(d.test, fault_severity = -1)) %>% arrange(id)

trainindex <- data.table(id = sort(unique(d.train$id)))
trainindex$index <- seq_along(trainindex$id)

testindex <- data.table(id = sort(unique(d.test$id)))
testindex$index <- seq_along(testindex$id)

totrainindex <- function(id) {
    map(id, from = trainindex$id, to = trainindex$index, warn_missing = FALSE)
}

totestindex <- function(id) {
    map(id, from = testindex$id, to = testindex$index, warn_missing = FALSE)
}

d.event$event_type <- as.integer(d.event$event_type)
d.feature$log_feature <- as.integer(d.feature$log_feature)
d.resource$resource_type <- as.integer(d.resource$resource_type)
d.severity$severity_type <- as.integer(d.severity$severity_type)

# Only allow locations in d.test.location which are present in the training set
d.location <- dplyr::select(d.set, id, location) %>% filter(location %in% d.set$location[d.set$fault_severity != -1])

d.location$location <- as.integer(d.location$location)
d.train.location <- d.location[d.location$id %in% d.train$id]
d.test.location <- d.location[d.location$id %in% d.test$id]

## d.event
# event_type 16 is not present in the dataset
s.event <- sparseMatrix(i = d.event$id[], j = d.event$event_type, x = 1)

s.event.train <- sparseMatrix(i = totrainindex(d.event$id[d.event$id %in% d.train$id]),
                              j = d.event$event_type[d.event$id %in% d.train$id],
                              x = 1)

s.event.test <- sparseMatrix(i = totestindex(d.event$id[d.event$id %in% d.test$id]),
                              j = d.event$event_type[d.event$id %in% d.test$id],
                              x = 1)

dim(s.event)
dim(s.event.train)
length(unique(d.event$id))
length(unique(d.event$event_type))

## d.feature
s.feature <- sparseMatrix(i = d.feature$id, j = d.feature$log_feature, x = d.feature$volume)

s.feature.train <- sparseMatrix(i = totrainindex(d.feature$id[d.feature$id %in% d.train$id]),
                                j = d.feature$log_feature[d.feature$id %in% d.train$id],
                                x = d.feature$volume[d.feature$id %in% d.train$id])

s.feature.test <- sparseMatrix(i = totestindex(d.feature$id[d.feature$id %in% d.test$id]),
                                j = d.feature$log_feature[d.feature$id %in% d.test$id],
                                x = d.feature$volume[d.feature$id %in% d.test$id])

dim(s.feature)
dim(s.feature.train)
length(unique(d.feature$id))
length(unique(d.feature$log_feature))

## d.resource
s.resource <- sparseMatrix(i = d.resource$id, j = d.resource$resource_type, x = 1)

s.resource.train <- sparseMatrix(i = totrainindex(d.resource$id[d.resource$id %in% d.train$id]),
                                 j = d.resource$resource_type[d.resource$id %in% d.train$id],
                                 x = 1)

s.resource.test <- sparseMatrix(i = totestindex(d.resource$id[d.resource$id %in% d.test$id]),
                                 j = d.resource$resource_type[d.resource$id %in% d.test$id],
                                 x = 1)

dim(s.resource)
dim(s.resource.train)
length(unique(d.resource$id))
length(unique(d.resource$resource_type))

## d.severity
# severity_type may or may not be ordinal
s.severity <- sparseMatrix(i = d.severity$id, j = d.severity$severity_type)

s.severity.train <- sparseMatrix(i = totrainindex(d.severity$id[d.severity$id %in% d.train$id]),
                           j = d.severity$severity_type[d.severity$id %in% d.train$id],
                           x = 1)

s.severity.test <- sparseMatrix(i = totestindex(d.severity$id[d.severity$id %in% d.test$id]),
                                 j = d.severity$severity_type[d.severity$id %in% d.test$id],
                                 x = 1)

dim(s.severity)
dim(s.severity.train)
length(unique(d.severity$id))
length(unique(d.severity$severity_type))

## d.train.location
s.location.train <- sparseMatrix(i = totrainindex(d.train.location$id),
                                 j = d.train.location$location,
                                 x = 1)

dim(s.location.train)
length(unique(d.train.location$id))
length(unique(d.train.location$location))

## d.location (test)
s.location.test <- sparseMatrix(i = totestindex(d.test.location$id),
                                j = d.test.location$location,
                                x = 1)

dim(s.location.test)
length(unique(d.test.location$id))
length(unique(d.test.location$location))

## y
y <- d.train$fault_severity

## xgb
# Get the feature 'real' names
names <- c(paste0("event", 1:15), paste0("event", 17:54),
           paste0("feature", 1:386),
           paste0("resource", 1:10),
           paste0("severity", 1:5),
           paste0("location", 1:929))

train.data = cBind(s.event.train, s.feature.train, s.resource.train, s.severity.train, s.location.train)
test.data = cBind(s.event.test, s.feature.test, s.resource.test, s.severity.test, s.location.test)

attr(train.data, "Dimnames")[[2]] <- names
attr(test.data, "Dimnames")[[2]] <- names

dtrain <- xgb.DMatrix(data = train.data, label = y)

num.class <- length(unique(d.train$fault_severity))

param <- list("objective" = "multi:softprob",    # multiclass classification
              "num_class" = num.class,    # number of classes
              "eval_metric" = "mlogloss",    # evaluation metric
              "nthread" = 8,   # number of threads to be used
              "max_depth" = 10,    # maximum depth of tree
              "eta" = 0.3,    # step size shrinkage
              "gamma" = 0,    # minimum loss reduction
              "subsample" = 1,    # part of data instances to grow tree
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree
              "min_child_weight" = 2,  # minimum sum of instance weight needed in a child
              "verbose" = 2
)

set.seed(1234)

cv.nround <- 45
cv.nfold <- 10

bst.cv <- xgb.cv(param=param, data=dtrain,
                 nfold=cv.nfold, nrounds=cv.nround, prediction=TRUE)

tail(bst.cv$dt)

# Index of minimum merror
min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean])

# sink(file = logfile, append = TRUE)
print(paste0("Min Error Index = ", min.error.index))

# Minimum error
print(bst.cv$dt[min.error.index, ])
# sink()

nround = min.error.index # number of trees generated
bst <- xgboost(param = param, data = dtrain, nrounds = nround, verbose = TRUE)

model <- xgb.dump(bst, with.stats = T)
model[1:10]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:20,])

# Tree plot - not working
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)

# Prediction
pred <- predict(bst, test.data)

# Decode prediction
pred <- matrix(pred, nrow=num.class, ncol=length(pred) / num.class)
pred <- data.frame(cbind(testindex$id, t(pred)))

plot(density(pred$predict_0), col = "green", ylim = c(0, 30))
lines(density(pred$predict_1), col = "orange")
lines(density(pred$predict_2), col = "red")

# You can dump the tree you learned using xgb.dump into a text file
xgb.dump(bst, "dump.raw.txt", with.stats = T)

# Finally, you can check which features are the most important.
print("Most important features (look at column Gain):")
imp_matrix <- xgb.importance(feature_names = names, filename_dump = "dump.raw.txt")
print(imp_matrix)

# output
filename <- "telstra_xgboost3.csv"
names(pred) <- c("id", "predict_0", "predict_1", "predict_2")
write.table(format(pred, scientific = FALSE), paste("./output/", filename, sep = ""), row.names = FALSE, sep = ",", quote = FALSE)

save.image("xgbimage.RData")

time.end <- Sys.time()
time.end - time.start

## glmnet.cr - need full rank, etc.
library(glmnetcr)

load("xgbimage.RData")
p <- as.data.frame(as.matrix(train.data))
r <- as.data.frame(as.matrix(test.data))
y.factor <- factor(y, levels = c(0, 1, 2), ordered = TRUE)

set.seed(1234)

fit <- glmnet.cr(train.data, y.factor)
print(fit)

BIC.step <- dplyr::select.glmnet.cr(fit)
BIC.step

# AIC.step <- dplyr::select.glmnet.cr(fit, which = "AIC")
# AIC.step
#
# coefficients <- coef(fit, s = BIC.step)
# coefficients$a0
#
# sum(coefficients$beta != 0)
#
nonzero.glmnet.cr(fit, s = BIC.step)
#
# plot(fit)


f <- fitted.glmnet.cr(object = fit, newx = test.data, s = AIC.step)
pred.glm <- data.frame(cbind(testindex$id, f$probs))


# output
filename <- "telstra_glmnetcr.csv"
names(pred.glm) <- c("id", "predict_0", "predict_1", "predict_2")
write.table(format(pred.glm, scientific = FALSE), paste("./output/", filename, sep = ""), row.names = FALSE, sep = ",", quote = FALSE)

# # Is every id present in the datasets?
# id.event <- unique(d.event$id)
# id.feature <- unique(d.feature$id)
# id.resource <- unique(d.feature$id)
# id.severity <- unique(d.severity$id)
# id.set <- sort(unique(c(d.train$id, d.test$id)))
#
# all.equal(id.event, id.feature)
# all.equal(id.event, id.resource)
# all.equal(id.event, id.severity)
# all.equal(id.event, id.set)
# all.equal(id.event, c(7,8,9))

# # count of different ids in d.event per event_type
# t1 <- aggregate(d.event$id ~ d.event$event_type, FUN = length)
# hist(t1[[2]], xlab = "Count of events per event_type")
#
# # count of different event types in d.event per id
# t2 <- aggregate(d.event$event_type ~ d.event$id, FUN = length)
# hist(t2[[2]], breaks = c(1:11), xlab = "Number of events per id")
# rm(t1, t2)



## Caret
set.seed(1234)

library(corrplot)
library(mlbench)
library(caret)

# check for zero variances
zero.var = nearZeroVar(train.data, saveMetrics=TRUE)
zero.var
zero.var[zero.var$zeroVar == TRUE, ]
zero.var[zero.var$nzv == FALSE, ]

cols <- row.names(zero.var[zero.var$nzv == TRUE, ]) # columns to discard
colNums <- match(cols, names(train))
ntrain <- dplyr::select(train, -colNums)

corr <- cor(ntrain)
corr

# correlation matrix
corrplot.mixed(cor(p), lower="circle", upper="color",
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")

## tsne plot
# t-Distributed Stochastic Neighbor Embedding
tsne = Rtsne(as.matrix(train), check_duplicates=FALSE, pca=TRUE,
             perplexity=30, theta=0, a <- c.5, dims=2)

embedding = as.data.frame(tsne$Y)
embedding$Class = outcome.org

g = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding of 'Classe' Outcome") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank())

print(g)

# pca see factominer,  clusterboost

# Confusion matrix - needs checking
#cv prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")

#Confusion matrix
confusionMatrix(factor(pred.cv), factor(y + 1))


### Naive Bayes
library(klaR)
nb <- NaiveBayes(p, y.factor)

# ------------------------------------------------------------
# 0. Load Data + Basic Cleaning
# ------------------------------------------------------------

raw_data <- `2020_bn_nb_data`
raw_data <- as.data.frame(raw_data)

# Convert all columns into factors (clean + consistent)
raw_data[] <- lapply(raw_data, function(col) factor(as.character(col)))

library(bnlearn)
library(e1071)

# ------------------------------------------------------------
# 1. Build Bayesian Network for Course Variables (V1–V8)
# ------------------------------------------------------------

# V9 = Internship outcome → target variable, remove it here
course_cols <- setdiff(names(raw_data), "V9")
course_only <- raw_data[, course_cols]

set.seed(123)
structure_courses <- hc(course_only)   # hill-climb structure learning
fitted_course_bn <- bn.fit(structure_courses, course_only)

# Print CPTs for each node
for (node_name in names(fitted_course_bn)) {
  cat("\nCPT of", node_name, ":\n")
  print(fitted_course_bn[[node_name]]$prob)
}

# ------------------------------------------------------------
# 2. Query Example:
#    PH100 grade (V6) given EC100=DD, IT101=CC, MA101=CD
# ------------------------------------------------------------

set.seed(2025)

query_sample <- cpdist(
  fitted_course_bn,
  nodes = "V6",
  evidence = (V1 == "DD" & V3 == "CC" & V5 == "CD"),
  n = 20000
)

cat("\nEstimated probability distribution for PH100 (V6):\n")
print(prop.table(table(query_sample$V6)))

# ------------------------------------------------------------
# 3. Naive Bayes Classification for Internship (V9)
# ------------------------------------------------------------

set.seed(42)
num_loops <- 20
nb_scores <- numeric(num_loops)

for (k in 1:num_loops) {
  
  shuffle_ids <- sample.int(nrow(raw_data))
  cutoff <- floor(0.7 * nrow(raw_data))
  
  train_set <- raw_data[shuffle_ids[1:cutoff], ]
  test_set  <- raw_data[shuffle_ids[(cutoff + 1):nrow(raw_data)], ]
  
  nb_model <- naiveBayes(V9 ~ ., data = train_set)
  nb_pred <- predict(nb_model, test_set)
  
  nb_scores[k] <- mean(nb_pred == test_set$V9)
  cat("NB Iteration", k, ": Accuracy =", nb_scores[k], "\n")
}

cat("\nNaive Bayes Average Accuracy:",
    mean(nb_scores), " | SD =", sd(nb_scores), "\n")

# ------------------------------------------------------------
# 4. Bayesian Network Classifier (Structure + CPT learned)
# ------------------------------------------------------------

bn_scores <- numeric(num_loops)

for (k in 1:num_loops) {
  
  shuffle_ids <- sample.int(nrow(raw_data))
  cutoff <- floor(0.7 * nrow(raw_data))
  
  train_bn <- raw_data[shuffle_ids[1:cutoff], ]
  test_bn  <- raw_data[shuffle_ids[(cutoff + 1):nrow(raw_data)], ]

  # Align factor levels across train/test
  for (col_name in names(raw_data)) {
    levs <- levels(raw_data[[col_name]])
    train_bn[[col_name]] <- factor(train_bn[[col_name]], levels = levs)
    test_bn[[col_name]]  <- factor(test_bn[[col_name]], levels = levs)
  }
  
  # Learn BN from training data only
  learned_struct <- hc(train_bn)
  fitted_bn_model <- bn.fit(learned_struct, train_bn)
  
  # Predict internship label V9
  pred_bn <- predict(
    fitted_bn_model,
    node = "V9",
    data = test_bn,
    method = "bayes-lw",
    n = 2000
  )
  
  bn_scores[k] <- mean(pred_bn == test_bn$V9)
  cat("BN Iteration", k, ": Accuracy =", bn_scores[k], "\n")
}

cat("\nBN Classifier Average Accuracy:",
    mean(bn_scores), " | SD =", sd(bn_scores), "\n")

# set wd
setwd("")

# You can install the sparklyr package from CRAN as follows:
# install.packages("sparklyr")

# You should also install a local version of Spark for development purposes:
library(sparklyr)
spark_install(version = "2.4.0")

# To upgrade to the latest version of sparklyr, run the following command and restart your r session:
# devtools::install_github("rstudio/sparklyr")

# Connecting to Spark
# You can connect to both local instances of Spark as well as remote Spark clusters. Here we’ll connect to a local instance of Spark via the spark_connect function:
sc <- spark_connect(master = "local")
# The returned Spark connection (sc) provides a remote dplyr data source to the Spark cluster.

# Using dplyr
# We can now use all of the available dplyr verbs against the tables within the cluster.
# We’ll start by copying some datasets from R into the Spark cluster (note that you may need to install the nycflights13 and Lahman packages in order to execute this code):
# install.packages(c("nycflights13", "Lahman"))

library(dplyr)
iris_tbl <- copy_to(sc, iris, overwrite = TRUE)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights", overwrite = TRUE)
batting_tbl <- copy_to(sc, Lahman::Batting, "batting", overwrite = TRUE)
src_tbls(sc)

# To start with here’s a simple filtering example:
# filter by departure delay and print the first few records
flights_tbl %>% filter(dep_delay == 2)

# Introduction to dplyr provides additional dplyr examples you can try. For example, consider the last example from the tutorial which plots data on flight delays:
delay <- flights_tbl %>%
  group_by(tailnum) %>%
  summarise(
    count = n(),
    dist = mean(distance),
    delay = mean(arr_delay)
  ) %>%
  filter(count > 20, dist < 2000,!is.na(delay)) %>%
  collect

# plot delays
library(ggplot2)
ggplot(delay, aes(dist, delay)) +
  geom_point(aes(size = count), alpha = 1 / 2) +
  geom_smooth() +
  scale_size_area(max_size = 2)
## `geom_smooth()` using method = 'gam'

# WINDOW FUNCTIONS
# dplyr window functions are also supported, for example:
batting_tbl %>%
  select(playerID, yearID, teamID, G, AB:H) %>%
  arrange(playerID, yearID, teamID) %>%
  group_by(playerID) %>%
  filter(min_rank(desc(H)) <= 2 & H > 0)

# Using SQL
# It’s also possible to execute SQL queries directly against tables within a Spark cluster. The spark_connection object implements a DBI interface for Spark, so you can use dbGetQuery to execute SQL and return the result as an R data frame:
library(DBI)
iris_preview <- dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview


# Machine Learning

# You can orchestrate machine learning algorithms in a Spark cluster via the machine learning functions within sparklyr. These functions connect to a set of high-level APIs built on top of DataFrames that help you create and tune machine learning workflows.

# Here’s an example where we use ml_linear_regression to fit a linear regression model. We’ll use the built-in mtcars dataset, and see if we can predict a car’s fuel consumption (mpg) based on its weight (wt), and the number of cylinders the engine contains (cyl). We’ll assume in each case that the relationship between mpg and each of our features is linear.

# copy mtcars into spark
mtcars_tbl <- copy_to(sc, mtcars)

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  mutate(cyl8 = cyl == 8) %>%
  sdf_partition(training = 0.5, test = 0.5, seed = 1099)

# fit a linear model to the training dataset
fit <- partitions$training %>%
  ml_linear_regression(response = "mpg", features = c("wt", "cyl"))
fit

# For linear regression models produced by Spark, we can use summary() to learn a bit more about the quality of our fit, and the statistical significance of each of our predictors.

summary(fit)

# Spark machine learning supports a wide array of algorithms and feature transformations and as illustrated above it’s easy to chain these functions together with dplyr pipelines. To learn more see the machine learning section.


# Reading and Writing Data

# You can read and write data in CSV, JSON, and Parquet formats. Data can be stored in HDFS, S3, or on the local filesystem of cluster nodes.

temp_csv <- tempfile(fileext = ".csv")
temp_parquet <- tempfile(fileext = ".parquet")
temp_json <- tempfile(fileext = ".json")

spark_write_csv(iris_tbl, temp_csv)
iris_csv_tbl <- spark_read_csv(sc, "iris_csv", temp_csv)

spark_write_parquet(iris_tbl, temp_parquet)
iris_parquet_tbl <- spark_read_parquet(sc, "iris_parquet", temp_parquet)

spark_write_json(iris_tbl, temp_json)
iris_json_tbl <- spark_read_json(sc, "iris_json", temp_json)

src_tbls(sc)

# Distributed R
# You can execute arbitrary r code across your cluster using spark_apply. For example, we can apply rgamma over iris as follows:
  
spark_apply(iris_tbl, function(data) {
  data[1:4] + rgamma(1,2)
})

# sdf_len(sc, 10) %>% spark_apply(function(df) df * 10)
# You can also group by columns to perform an operation over each group of rows and make use of any package within the closure:
  
spark_apply(
  iris_tbl,
  function(e) broom::tidy(lm(Petal_Width ~ Petal_Length, e)),
  names = c("term", "estimate", "std.error", "statistic", "p.value"),
  group_by = "Species"
)


# Extensions
# The facilities used internally by sparklyr for its dplyr and machine learning interfaces are available to extension packages. Since Spark is a general purpose cluster computing system there are many potential applications for extensions (e.g. interfaces to custom machine learning pipelines, interfaces to 3rd party Spark packages, etc.).

# Here’s a simple example that wraps a Spark text file line counting function with an R function:
  
# write a CSV 
tempfile <- tempfile(fileext = ".csv")
write.csv(nycflights13::flights, tempfile, row.names = FALSE, na = "")

# define an R interface to Spark line counting
count_lines <- function(sc, path) {
  spark_context(sc) %>% 
    invoke("textFile", path, 1L) %>% 
    invoke("count")
}

# call spark to count the lines of the CSV
count_lines(sc, tempfile)

# Table Utilities
# You can cache a table into memory with:
  
tbl_cache(sc, "batting")
# and unload from memory using:
  
tbl_uncache(sc, "batting")
# Connection Utilities
# You can view the Spark web console using the spark_web function:
  
spark_web(sc)
# You can show the log using the spark_log function:
  
spark_log(sc, n = 10)
# Finally, we disconnect from Spark:

spark_disconnect(sc)


# Using H2O
# rsparkling is a CRAN package from H2O that extends sparklyr to provide an interface into Sparkling Water. For instance, the following example installs, configures and runs h2o.glm:
options(rsparkling.sparklingwater.version = "2.4.1")

library(rsparkling)
library(sparklyr)
library(dplyr)
library(h2o)
# devtools::install_version("h2o", version = "3.22.0.3", repos = "http://cran.us.r-project.org")

sc <- spark_connect(master = "local", version = "2.4.0")
mtcars_tbl <- copy_to(sc, mtcars, "mtcars")

mtcars_h2o <- as_h2o_frame(sc, mtcars_tbl, strict_version_check = FALSE)

mtcars_glm <- h2o.glm(x = c("wt", "cyl"), 
                      y = "mpg",
                      training_frame = mtcars_h2o,
                      lambda_search = TRUE)
mtcars_glm

spark_disconnect(sc)
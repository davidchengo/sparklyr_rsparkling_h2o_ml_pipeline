# set wd
setwd("")

# Spark’s ML Pipelines provide a way to easily combine multiple transformations and algorithms into a single workflow, or pipeline.

# For R users, the insights gathered during the interactive sessions with Spark can now be converted to a formal pipeline. This makes the hand-off from Data Scientists to Big Data Engineers a lot easier, this is because there should not be additional changes needed to be made by the later group.
 
# The final list of selected variables, data manipulation, feature transformations and modeling can be easily re-written into a ml_pipeline() object, saved, and ultimately placed into a Production environment. The sparklyr output of a saved Spark ML Pipeline object is in Scala code, which means that the code can be added to the scheduled Spark ML jobs, and without any dependencies in R.

 
# Introduction to ML Pipelines
# The official Apache Spark site contains a more complete overview of ML Pipelines. This article will focus in introducing the basic concepts and steps to work with ML Pipelines via sparklyr.

# There are two important stages in building an ML Pipeline. The first one is creating a Pipeline. A good way to look at it, or call it, is as an “empty” pipeline. This step just builds the steps that the data will go through. This is the somewhat equivalent of doing this in R:
r_pipeline <-  . %>% mutate(cyl = paste0("c", cyl)) %>% lm(am ~ cyl + mpg, data = .)
r_pipeline

# The r_pipeline object has all the steps needed to transform and fit the model, but it has not yet transformed any data.

# The second step, is to pass data through the pipeline, which in turn will output a fitted model. That is called a PipelineModel. The PipelineModel can then be used to produce predictions.

r_model <- r_pipeline(mtcars)
r_model

# TAKING ADVANTAGE OF PIPELINES AND PIPELINEMODELS
# The two stage ML Pipeline approach produces two final data products:

# A PipelineModel that can be added to the daily Spark jobs which will produce new predictions for the incoming data, and again, with no R dependencies.

# A Pipeline that can be easily re-fitted on a regular interval, say every month. All that is needed is to pass a new sample to obtain the new coefficients.

# Pipeline
# An additional goal of this article is that the reader can follow along, so the data, transformations and Spark connection in this example will be kept as easy to reproduce as possible.

library(nycflights13)
library(sparklyr)
library(dplyr)
sc <- spark_connect(master = "local", spark_version = "2.2.0")

## * Using Spark: 2.2.0

spark_flights <- sdf_copy_to(sc, flights)

# FEATURE TRANSFORMERS
# Pipelines make heavy use of Feature Transformers. If new to Spark, and sparklyr, it would be good to review what these transformers do. These functions use the Spark API directly to transform the data, and may be faster at making the data manipulations that a dplyr (SQL) transformation.

# In sparklyr the ft functions are essentially are wrappers to original Spark feature transformer.

# FT_DPLYR_TRANSFORMER
# This example will start with dplyr transformations, which are ultimately SQL transformations, loaded into the df variable.

# In sparklyr, there is one feature transformer that is not available in Spark, ft_dplyr_transformer(). The goal of this function is to convert the dplyr code to a SQL Feature Transformer that can then be used in a Pipeline.

df <- spark_flights %>%
  filter(!is.na(dep_delay)) %>%
  mutate(
    month = paste0("m", month),
    day = paste0("d", day)
  ) %>%
  select(dep_delay, sched_dep_time, month, day, distance) 

# This is the resulting pipeline stage produced from the dplyr code:
ft_dplyr_transformer(sc, df)

# Use the ml_param() function to extract the “statement” attribute. That attribute contains the finalized SQL statement. Notice that the flights table name has been replace with __THIS__. This allows the pipeline to accept different table names as its source, making the pipeline very modular.

ft_dplyr_transformer(sc, df) %>%
  ml_param("statement")

# CREATING THE PIPELINE
# The following step will create a 5 stage pipeline:

# 1. SQL transformer - Resulting from the ft_dplyr_transformer() transformation
# 2. Binarizer - To determine if the flight should be considered delay. The eventual outcome variable.
# 3. Bucketizer - To split the day into specific hour buckets
# 4. R Formula - To define the model’s formula
# 5. Logistic Model

flights_pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(
    tbl = df
  ) %>%
  ft_binarizer(
    input.col = "dep_delay",
    output.col = "delayed",
    threshold = 15
  ) %>%
  ft_bucketizer(
    input.col = "sched_dep_time",
    output.col = "hours",
    splits = c(400, 800, 1200, 1600, 2000, 2400)
  )  %>%
  ft_r_formula(delayed ~ month + day + hours + distance) %>% 
  ml_logistic_regression()

# Another nice feature for ML Pipelines in sparklyr, is the print-out. It makes it really easy to how each stage is setup:
flights_pipeline

# Notice that there are no coefficients defined yet. That’s because no data has been actually processed. Even though df uses spark_flights(), recall that the final SQL transformer makes that name, so there’s no data to process yet.

# PipelineModel
# A quick partition of the data is created for this exercise.

partitioned_flights <- sdf_partition(
  spark_flights,
  training = 0.01,
  testing = 0.01,
  rest = 0.98
)
# The ml_fit() function produces the PipelineModel. The training partition of the partitioned_flights data is used to train the model:
  
fitted_pipeline <- ml_fit(
  flights_pipeline,
  partitioned_flights$training
)
fitted_pipeline

# Notice that the print-out for the fitted pipeline now displays the model’s coefficients.

# The ml_transform() function can be used to run predictions, in other words it is used instead of predict() or sdf_predict().

predictions <- ml_transform(
  fitted_pipeline,
  partitioned_flights$testing
)

predictions %>%
  group_by(delayed, prediction) %>%
  tally()


# Save the pipelines to disk
# The ml_save() command can be used to save the Pipeline and PipelineModel to disk. The resulting output is a folder with the selected name, which contains all of the necessary Scala scripts:
  
ml_save(
  flights_pipeline,
  "flights_pipeline",
  overwrite = TRUE
)

## NULL

ml_save(
  fitted_pipeline,
  "flights_model",
  overwrite = TRUE
)

## NULL


# Use an existing PipelineModel
# The ml_load() command can be used to re-load Pipelines and PipelineModels. The saved ML Pipeline files can only be loaded into an open Spark session.

reloaded_model <- ml_load(sc, "flights_model")
# A simple query can be used as the table that will be used to make the new predictions. This of course, does not have to done in R, at this time the “flights_model” can be loaded into an independent Spark session outside of R.

new_df <- spark_flights %>%
  filter(
    month == 7,
    day == 5
  )

ml_transform(reloaded_model, new_df) 

# Re-fit an existing Pipeline
# First, reload the pipeline into an open Spark session:
  
reloaded_pipeline <- ml_load(sc, "flights_pipeline")

# Use ml_fit() again to pass new data, in this case, sample_frac() is used instead of sdf_partition() to provide the new data. The idea being that the re-fitting would happen at a later date than when the model was initially fitted.

new_model <-  ml_fit(reloaded_pipeline, sample_frac(spark_flights, 0.01))

new_model

# The new model can be saved using ml_save(). A new name is used in this case, but the same name as the existing PipelineModel to replace it.

ml_save(new_model, "new_flights_model", overwrite = TRUE)

## NULL
# Finally, this example is complete by closing the Spark session.

spark_disconnect(sc)

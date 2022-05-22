# item-classifier-python

A reusable library for classifying different objects

## Install python requirements with Anaconda

1. `conda create -n tf python=3.8.8`
2. `conda activate tf`
3. cd into the repository
4. `pip install -r requirements.txt`

## Getting the tick dataset

1. Obtain your dataset.
2. Extract the `dataset.zip` into `src/tf_files/dataset`
3. rename all files to .jpg with `dir | Rename-Item -NewName { $_.name -replace ".PNG",".jpg"}`

## Retrain and Conversion

1. `(Optional)` Start tensorboard

```{python}
tensorboard --logdir tf_files/training_summaries &
```

#### note that the "`^`" is an escape character and may vary with the terminal you're using. This works for command prompt in windows.

2. Run retrain script

```{python}
python -m scripts.retrain ^
  --image_dir=tf_files/dataset ^
  --model_dir=tf_files/models ^
  --architecture=mobilenet_0.50_224 ^
  --output_graph=tf_files/model_graph.pb ^
  --output_labels=tf_files/model_labels.txt ^
  --bottleneck_dir=tf_files/bottlenecks ^
  --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 ^
  --how_many_training_steps=400 ^
  --learning_rate=0.001
```

3. `(Optional)` Quantize the graph for better performance in javascript

```{python}
python -m scripts.quantize_graph ^
  --input=tf_files/model_graph.pb ^
  --output=tf_files/quantized_model_graph.pb ^
  --output_node_names=final_result ^
  --mode=weights_rounded
```

4. Convert model into tensorflowjs format

```{python}
tensorflowjs_converter ^
  --input_format=tf_frozen_model ^
  --output_node_names=final_result ^
  tf_files/quantized_model_graph.pb ^
  tf_files/web
```

## Prediction

```{python}
python -m scripts.predict ^
  --image=tf_files/dataset/your_image.jpg ^
  --labels=tf_files/model_labels.txt ^
  --graph=tf_files/model_graph.pb
```

The converted tensorflowjs model is in `tf_files/web` directory

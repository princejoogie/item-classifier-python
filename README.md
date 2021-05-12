# Tick Classifier Python Training

## Setup Anaconda

1. `conda create -n tf python=3.8.8`
2. `conda activate tf`
3. cd into the repository
4. `pip install -r requirements.txt`

## Usage

1. Download dataset with from [here](https://joog.uno/ticks_ds1)
2. Extract the `tick_photos.zip` into `src/tf_files/tick_photos`
3. Open `retrain_cmd.txt` to see available commands.

## Retrain and Prediction

1. Start tensorboard

```{python}
tensorboard --logdir tf_files/training_summaries &
```

#### note that the "`^`" is an escape character and may vary with the terminal you're using. This works for command prompt in windows.

2. Run retrain script

```{python}
python -m scripts.retrain ^
  --image_dir=tf_files/tick_photos ^
  --model_dir=tf_files/models ^
  --architecture=mobilenet_0.50_224 ^
  --output_graph=tf_files/tick_graph.pb ^
  --output_labels=tf_files/tick_labels.txt ^
  --bottleneck_dir=tf_files/bottlenecks ^
  --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 ^
  --how_many_training_steps=400 ^
  --learning_rate=0.00
```

3. `(Optional)` Quantize the graph for better performance in javascript

```{python}
python -m scripts.quantize_graph ^
  --input=tf_files/tick_graph.pb ^
  --output=tf_files/tick_graph.pb ^
  --output_node_names=final_result ^
  --mode=weights_rounded
```

4. Convert model into tensorflowjs format

```{python}
tensorflowjs_converter ^
  --input_format=tf_frozen_model ^
  --output_node_names=final_result ^
  tf_files/tick_graph.pb ^
  tf_files/web
```

5. Run Prediction script

```{python}
python -m scripts.predict ^
  --image=tf_files/tick_photos/brown_tick/27.PNG ^
  --labels=tf_files/tick_labels.txt ^
  --graph=tf_files/tick_graph.pb
```

The converted tensorflowjs model is in `tf_files/web` directory

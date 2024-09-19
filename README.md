# New York City - Taxi Fare Prediction

## Kaggle competition, dataset

[Project link](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)

## Goal

Create a regression model that can predict the fare of a taxi trip.

## Methodology

### Dealing with the massive size of the dataset

This dataset is **HUUUUUUGGEE**, so I knew I had to use Polars and Tensorflow, both with GPU support to train a model on.

The dataset is so huge, that it was larger than the VRAM on my GPU, so the kernel kept dying. I had to limit myself to only 1,000,000 rows, of of the total 55,000,000 (I probably could have used 30,000,000 rows).

```python
df = pl.read_csv("../../datasets/new-york-city-taxi-fare-prediction/train.csv", n_rows=1_000_000)
```

### Creating the model

Since the dataset is so large (larger than any dataset I have ever made a model on, besides MNIST), I decided to keep the model architecture easy. It is a simple Pass Forwards model with `BatchNormalization()` and `Dropout()`.

### Compiling and fitting the model

The dataset trained a lot of epochs rather quickly, and I wanted to see the evaluation metrics of each epoch. But there were 300 epochs, and so to avoid my IDE from being overloaded with print statements, I created a EpochLogger class, that would only print the progress every 10 epochs. You just pass this in the `callback=[]` part of the model-fit.

I also used `early_stopping`, with `restore_best_weights=True`

### Creating submission

After training the model, I just had to make predictions on the test dataset, and export the necessary columns to a CSV and hand it in on Kaggle.

Here is an overview of my submissions:

| attempt | Public score | MSE                | MAE                |
| ------- | ------------ | ------------------ | ------------------ |
| 1       | 18.18691     | didn't measure     | didn't measure     |
| 2       | 15.18163     | 233.9773221521298  | 14.162008458814446 |
| 3       | 12.79653     | 168.16301793033367 | 11.639570094612424 |

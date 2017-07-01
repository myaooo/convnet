
## Data Prerequisite

Download data from Kaggle and placed the unzipped data under the `data/fer2013` directory of this project.

## Run

run the following script under project root
```python
python -m cnn.model_face --model=4 --name=CustomModelName --epoch=65 --task=train,test
```
Should be able to achieve 67% accuracy on test set.

## Options


### `--model`

`--model=1`: A test Network in Network model

`--model=2`: A NIN model

`--model=3`: A residual network without bottleneck

`--model=4`: A residual network with bottleneck

### `--epoch`

The epoch number to run

### `--name`

A unique name as an identifier for the model. Duplicate name of existing model would override the saved model on disk

### `--task`

Specify tasks to run, default to `train,test`. If you only want to test an existing model, run with `--task=test`.


# forecasting-sots-experiments

This repository contains the Python code used to carry out the experimental work to compare different approaches to perform forecasts using Sequences of Time Series (SoTS).

The repository structure is as follows:
- `cfg`: In this folder is the configuration file.
- `data`: This folder contains the dataset.
- `src`: This folder contains the code for the experimental analysis.

The scripts used in the experimentation are the following:

- _main.py_: script that launches the experiments.
- _main_errors.py_: script that computes the errors of the experiments.
- _PLOT*.py_: scripts that generate the tables and visualisations related to the results or metadata of the data.

## Data

In order to carry out the experimentation, it is first necessary to have the battery data. The dataset can be downloaded from [here](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). The original files are in _mat_ format, and must first be transformed into _pickle_ format. The data for each battery must be in a single file in the path `data/cell`. An example is available in the `data_example` directory.

## Interpolation

The regularized SoTS values are obtained by linear interpolation. In order to be able to do that for all experiments, for each variable of each battery the interpolation functions are stored in a file, which are used to regularise the SoTS and to obtain the values of the TS at the selected timestamps. These files are stored in the path `interpolation`.

## Experimentation

The experimentation is run on the command line, launching the main.py script. The file needs 3 arguments. 
1. **Battery index**. In the original dataset, 129 batteries are available hence an index value ranging from 1 to 129 is expected.
2. **Variable index**. The index of the variable for which you want to make predictions in the SoTS within the set [_Voltage_, _Discharging capacity_, _Temperature_, _Internal resistance_, _Charging capacity_].
3. **Series length index**. In the experimental work this argument takes a value of [50, 100, 150], and this argument represents the index of the set of possible lengths of the TS.

An example of the framework execution:

```
python main.py 78 3 1
```

This combination of arguments equals battery number 78, temperature and length 50.

## Evaluation

Once the execution of `main.py` finishes, errors are computed by running the `main_errors.py` script on the command line. The execution is analogous to the experimental script.
Additionally, scripts with the suffix _PLOT_ have been used to generate the visualisations of the error distributions of the different models and the training time results.

## Support and Author

Amaia Arregi Egibar

aarregui@ideko.es

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://github.com/aarregie)

## Citation

If you find useful the code in your research, please include explicit mention of our work in your publication with the following corresponding entry in your bibliography:

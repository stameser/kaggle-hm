# Project setup

Create conda environment

`conda env create -f conda.yaml`

Install local package
```bash
conda activate kagglehm
pip install -e .
```

Create data directory

```bash
mkdir data
mkdir data/raw
mkdir data/clean
mdkir data/output
mkdir data/clean/simil

kaggle competitions download -c h-and-m-personalized-fashion-recommendations
# unzip and move data to data/raw
```



convert nb to html
`jupyter nbconvert --to html .\notebooks\example.ipynb --output=example.html`
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

# Training flow
```bash
python als_model.py candidates --min-items 2 --min-customers 2 --factors 256 --cell-value time --iterations 50 --model-date 2020-09-08 --recs 24
python ranking_features.py --prediction-date 2020-09-08
```

# Evaluation flow 
```bash
python als_model.py candidates --min-items 2 --min-customers 2 --factors 256 --cell-value time --iterations 50 --model-date 2020-09-15 --recs 24
python ranking_features.py --prediction-date 2020-09-15
```

# Subsmission flow
```bash
python als_model.py candidates --min-items 2 --min-customers 2 --factors 256 --cell-value time --iterations 50 --model-date 2020-09-22 --recs 24
python ranking_features.py --prediction-date 2020-09-22
```

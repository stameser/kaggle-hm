# Project setup

Create conda environment

`conda env create -f conda.yaml`

Install local package
```bash
conda activate kagglehm
pip install -e .
```

convert nb to html
`jupyter nbconvert --to html .\notebooks\example.ipynb --output=example.html`
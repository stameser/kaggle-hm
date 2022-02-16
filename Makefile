upload:
	kaggle competitions submit h-and-m-personalized-fashion-recommendations -f data/output/submission.csv

fit:
	dvc repro
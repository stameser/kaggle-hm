import os
from pathlib import Path

data_root = Path(os.environ.get('KAGGLE_HM_DATA', ""))
train_dates = {
    'start': '2020-08-01',
    'end': '2020-09-08'
}
test_dates = {
    'start': '2020-09-09',
    'end': '2020-09-15'
}

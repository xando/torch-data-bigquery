# Torch Data Bigquery

[![PyPI - Version](https://img.shields.io/pypi/v/torch-data-bigquery.svg)](https://pypi.org/project/torch-data-bigquery)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-data-bigquery.svg)](https://pypi.org/project/torch-data-bigquery)

-----

Torch `Dataset` interface for Google BigQuery datasets.


**Table of Contents**

- [Torch Data Bigquery](#torch-data-bigquery)
  - [Installation](#installation)
  - [Examples](#examples)
  - [License](#license)

## Installation

```console
pip install torch-data-bigquery
```

## Examples 

`BigQueryStorageDataset`

```python

import torch
from torch_data_bigquery import BigQueryStorageDataset

dataset = BigQueryStorageDataset(
    billing_project=PROJECT,
    selected_fields=[
        "PassengerId",
        "Survived",
        "Pclass",
    ],
    location=f"bq://your-gcp-project.kaggle_titanic.train",
)

dataloader = torch.utils.data.DataLoader(dataset)

```


`BigQueryDataset` 

```python

import torch
from torch_data_bigquery import BigQueryDataset

dataset = BigQueryDataset(
    billing_project=PROJECT,
    query=f"""
        SELECT
            Survived                                AS survived,    # 1
            Pclass                                  AS pclass,      # 2
            DENSE_RANK() OVER(ORDER by Sex)         AS sex,         # 3   
            COALESCE(Age, AVG(Age) OVER())          AS age,         # 4
            SibSp                                   AS siblings,    # 5
            Parch                                   AS parents,     # 6
            Fare                                    AS fare,        # 7
            DENSE_RANK() OVER(ORDER by Embarked)    AS embarked,    # 8
        FROM `your-gcp-project.kaggle_titanic.train`
    """,
)

dataloader = torch.utils.data.DataLoader(dataset)

```


## License

`torch-data-bigquery` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

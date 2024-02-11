import os

import torch
import pyarrow

from torch_data_bigquery import BigQueryDataset


PROJECT = os.environ["GCP_PROJECT"]


def test_simple_fetch():

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
        FROM `{PROJECT}.kaggle_titanic.train`
        """,
    )

    dataloader = torch.utils.data.DataLoader(dataset)

    data = list(dataloader)

    assert len(data) == 891
    assert len(data[0][0]) == 8



def test_with_mapping():

    def to_categorical(x):
        return torch.tensor(pyarrow.compute.equal(x, 'female').cast(pyarrow.int8()).to_numpy())

    dataset = BigQueryDataset(
        billing_project=PROJECT,
        query=f"""
            SELECT
                Survived                                AS survived,    # 1
                Pclass                                  AS pclass,      # 2
                Sex                                     AS sex,         # 3   
                COALESCE(Age, AVG(Age) OVER())          AS age,         # 4
                SibSp                                   AS siblings,    # 5
                Parch                                   AS parents,     # 6
                Fare                                    AS fare,        # 7
                DENSE_RANK() OVER(ORDER by Embarked)    AS embarked,    # 8
            FROM `{PROJECT}.kaggle_titanic.train`
        """,
        fields_transform={"sex": to_categorical}
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

    data = list(dataloader)

    categories, counts = data[0][:, 2].unique(return_counts=True)

    assert [0, 1] == categories.tolist()
    assert [577, 314] == counts.tolist()    

import os

import torch
import pyarrow


from torch_data_bigquery import BigQueryStorageDataset


PROJECT = os.environ["GCP_PROJECT"]


def test_simple_fetch():
    dataset = BigQueryStorageDataset(
        billing_project=PROJECT,
        selected_fields=[
            "PassengerId",
            "Survived",
            "Pclass",
        ],
        location=f"bq://{PROJECT}.kaggle_titanic.train",
    )

    dataloader = torch.utils.data.DataLoader(dataset)

    data = list(dataloader)

    assert len(data) == 891
    assert len(data[0][0]) == 3

    # to check if the data is being fetched correctly after reseting iteraor
    data = list(dataloader)

    assert len(data) == 891
    assert len(data[0][0]) == 3


def test_with_mapping():

    def to_categorical(x):
        return torch.tensor(pyarrow.compute.equal(x, 'female').cast(pyarrow.int8()).to_numpy())

    dataset = BigQueryStorageDataset(
        billing_project=PROJECT,
        selected_fields=[
            "PassengerId",
            "Survived",
            "Pclass",
            "Sex"
        ],
        location=f"bq://{PROJECT}.kaggle_titanic.train",
        fields_transform={"Sex": to_categorical}
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

    data = list(dataloader)

    categories, counts = data[0][:, 3].unique(return_counts=True)

    assert [0, 1] == categories.tolist()
    assert [577, 314] == counts.tolist()

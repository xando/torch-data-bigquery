from google.cloud import bigquery
from torch.utils.data import IterableDataset

from .storage import BigQueryStorageDataset


class BigQueryDataset(IterableDataset):

    def __init__(
        self,
        *,
        billing_project,
        query,
        fields_transform=None,
    ):
        super().__init__()

        self.billing_project = billing_project
        self.query = query
        self.fields_transform = fields_transform or {}

    def __iter__(self):
        query_client = bigquery.Client(project=self.billing_project)
        query_job = query_client.query(self.query)
        query_job.result()

        destination = query_job.destination

        location = f"bq://{destination.project}.{destination.dataset_id}.{destination.table_id}"

        self.dataset = BigQueryStorageDataset(
            billing_project=self.billing_project,
            location=location,
            fields_transform=self.fields_transform
        )
        return self.dataset.__iter__()

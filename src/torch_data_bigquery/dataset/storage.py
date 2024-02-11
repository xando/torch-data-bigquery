import multiprocessing
import queue
import os
import functools

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from urllib.parse import urlparse

import numpy
import pyarrow
from google.cloud import bigquery_storage
from torch.utils.data import IterableDataset


PYARROW_TO_NUMPY = functools.partial(pyarrow.Array.to_numpy, zero_copy_only=False)


def _generate_streams(
    parent: str,
    dataset_uri: str,
    selected_fields: List[str],
    row_restrictions: str,
    max_stream_count: int,
) -> Tuple[List[str], pyarrow.Schema]:

    uri = urlparse(dataset_uri)

    assert uri.scheme == "bq", f"Invalid scheme: {uri.scheme}, expected 'bq'"
    assert uri.path == "", f"Invalid path: {uri.path}, expected empty string"

    project, dataset, table = uri.netloc.split(".")

    read_session = bigquery_storage.ReadSession(
        table=f"projects/{project}/datasets/{dataset}/tables/{table}",
        data_format=bigquery_storage.DataFormat.ARROW,
        read_options={
            "selected_fields": selected_fields,
            "row_restriction": row_restrictions,
        },
    )

    client = bigquery_storage.BigQueryReadClient()
    read_session = client.create_read_session(
        parent=f"projects/{parent}",
        read_session=read_session,
        max_stream_count=max_stream_count,
    )

    schema_buffer = pyarrow.py_buffer(
        read_session.arrow_schema.serialized_schema
    )
    schema = pyarrow.ipc.read_schema(schema_buffer)

    return read_session.streams, schema


def _read_streams(schema, queue_streams, queue_results):
    client = bigquery_storage.BigQueryReadClient()

    try:
        while stream := queue_streams.get(block=False):
            for message in client.read_rows(stream.name):
                record_batch = pyarrow.ipc.read_record_batch(
                    message.arrow_record_batch.serialized_record_batch,
                    schema
                )
                queue_results.put(record_batch)
    except queue.Empty:
        queue_results.put(None)


class BigQueryStorageDataset(IterableDataset):

    def __init__(
        self,
        *,
        billing_project,
        location,
        row_restrictions=None,
        selected_fields=None,
        max_stream_count=os.cpu_count(),
        fields_transform=None,
    ):
        super().__init__()
        self.project = billing_project
        self.location = location
        self.selected_fields = selected_fields
        self.row_restrictions = row_restrictions
        self.max_stream_count = max_stream_count
        self.fields_transform = fields_transform or {}

    def __iter__(self):
        streams, streams_schema = _generate_streams(
            parent=self.project,
            dataset_uri=self.location,
            selected_fields=self.selected_fields,
            row_restrictions=self.row_restrictions,
            max_stream_count=self.max_stream_count,
        )

        mp_manager = multiprocessing.Manager()
        self.queue_streams = mp_manager.Queue()
        self.queue_results = mp_manager.Queue()

        for stream in streams:
            self.queue_streams.put(stream)

        max_workers = min(len(streams), self.max_stream_count)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.submit(
                _read_streams,
                streams_schema,
                self.queue_streams,
                self.queue_results,
            )

        done = 0
        while True:
            element = self.queue_results.get(block=True)
            if not element:
                done += 1
                if done == max_workers:
                    break

            data = []
            for e in element.column_names:
                field_transform = self.fields_transform.get(e, PYARROW_TO_NUMPY)
                data.append(field_transform(element[e]))

            np_data = numpy.array(data)

            for e in range(np_data.shape[1]):
                yield np_data[:, e]

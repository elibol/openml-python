import unittest
import os
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import openml
from openml import OpenMLDataset
from openml.util import is_string
from openml.testing import TestBase

from openml.datasets.functions import (_get_cached_dataset,
                                       _get_cached_datasets,
                                       _get_dataset_description,
                                       _get_dataset_features,
                                       _get_dataset_qualities)


class TestOpenMLDataset(TestBase):
    def test__get_cached_datasets(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        openml.config.set_cache_directory(cachedir=workdir, privatedir=workdir)
        datasets = _get_cached_datasets()
        self.assertIsInstance(datasets, dict)
        self.assertEqual(len(datasets), 2)
        self.assertIsInstance(list(datasets.values())[0], OpenMLDataset)

    def test__get_cached_dataset(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        openml.config.set_cache_directory(workdir, workdir)
        api_mock_return_value = 400, \
            """<oml:authenticate xmlns:oml = "http://openml.org/openml">
            <oml:session_hash>G9MPPN114ZCZNWW2VN3JE9VF1FMV8Y5FXHUDUL4P</oml:session_hash>
            <oml:valid_until>2014-08-13 20:01:29</oml:valid_until>
            <oml:timezone>Europe/Berlin</oml:timezone>
            </oml:authenticate>"""
        with mock.patch("openml._api_calls._perform_api_call",
                        return_value=api_mock_return_value) as api_mock:
            dataset = _get_cached_dataset(2)
            self.assertIsInstance(dataset, OpenMLDataset)
            self.assertTrue(api_mock.is_called_once())

    def test_get_chached_dataset_description(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        openml.config.set_cache_directory(workdir, workdir)
        description = openml.datasets.functions._get_cached_dataset_description(2)
        self.assertIsInstance(description, dict)

    def test_list_datasets(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        datasets = openml.datasets.list_datasets()
        # 1087 as the number of datasets on openml.org
        self.assertGreaterEqual(len(datasets), 1087)
        for dataset in datasets:
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertTrue(is_string(dataset['status']))
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])

    def test_list_datasets_by_tag(self):
        datasets = openml.datasets.list_datasets_by_tag('uci')
        self.assertGreaterEqual(len(datasets), 5)
        for dataset in datasets:
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertTrue(is_string(dataset['status']))
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])

    @unittest.skip("Not implemented yet.")
    def test_check_datasets_active(self):
        raise NotImplementedError()

    def test_get_datasets(self):
        dids = [1, 2]
        datasets = openml.datasets.get_datasets(dids)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "dataset.arff")))

    def test_get_dataset(self):
        dataset = openml.datasets.get_dataset(1)
        self.assertEqual(type(dataset), OpenMLDataset)
        self.assertEqual(dataset.name, 'anneal')
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "dataset.arff")))

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 164
        dataset = openml.datasets.get_dataset(did)
        self.assertEqual(dataset.row_id_attribute, 'instance')

    def test__get_dataset_description(self):
        # Only a smoke test, I don't know exactly how to test the URL
        # retrieval and "caching"
        description = _get_dataset_description(2)
        self.assertIsInstance(description, dict)

    def test__get_dataset_features(self):
        # Only a smoke check
        features = _get_dataset_features(2)
        self.assertIsInstance(features, dict)

    def test__get_dataset_qualities(self):
        # Only a smoke check
        qualities = _get_dataset_qualities(2)
        self.assertIsInstance(qualities, dict)

    def test_publish_dataset(self):

        dataset = openml.datasets.get_dataset(3)
        file_path = os.path.join(openml.config.get_cache_directory(),
                                 "datasets", "3", "dataset.arff")
        dataset = OpenMLDataset(
            name="anneal", version=1, description="test",
            format="ARFF", licence="public", default_target_attribute="class", data_file=file_path)
        return_code, return_value = dataset.publish()
        # self.assertTrue("This is a read-only account" in return_value)
        self.assertEqual(return_code, 200)

    def test_upload_dataset_with_url(self):
        dataset = OpenMLDataset(
            name="UploadTestWithURL", version=1, description="test",
            format="ARFF",
            url="http://expdb.cs.kuleuven.be/expdb/data/uci/nominal/iris.arff")
        return_code, return_value = dataset.publish()
        # self.assertTrue("This is a read-only account" in return_value)
        self.assertEqual(return_code, 200)

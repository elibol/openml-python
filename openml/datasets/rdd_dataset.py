import gzip
import os
import sys
import logging
import arff

import numpy as np
import scipy.sparse

from ..util import is_string

logger = logging.getLogger(__name__)


def create_basic_dataset(description, arff_string):
    """Create a dataset object from a description dict.

    Parameters
    ----------
    description : dict
        Description of a dataset in xmlish dict.
    arff_string : string
        contents of dataset arff file.

    Returns
    -------
    dataset : dataset object
        Dataset object from dict and arff.
    """
    dataset = BasicOpenMLDataset(
        description["oml:id"],
        description["oml:name"],
        description["oml:version"],
        description.get("oml:description"),
        description["oml:format"],
        description.get("oml:creator"),
        description.get("oml:contributor"),
        description.get("oml:collection_date"),
        description.get("oml:upload_date"),
        description.get("oml:language"),
        description.get("oml:licence"),
        description["oml:url"],
        description.get("oml:default_target_attribute"),
        description.get("oml:row_id_attribute"),
        description.get("oml:ignore_attribute"),
        description.get("oml:version_label"),
        description.get("oml:citation"),
        description.get("oml:tag"),
        description.get("oml:visibility"),
        description.get("oml:original_data_url"),
        description.get("oml:paper_url"),
        description.get("oml:update_comment"),
        description.get("oml:md5_checksum"),
        arff_string=arff_string)
    return dataset


class BasicOpenMLDataset(object):
    """Dataset object.

    Allows fetching and uploading datasets to OpenML.

    Parameters
    ----------
    name : string
        Name of the dataset
    description : string
        Description of the dataset
    FIXME : which of these do we actually nee?
    """
    def __init__(self, id=None, name=None, version=None, description=None,
                 format=None, creator=None, contributor=None,
                 collection_date=None, upload_date=None, language=None,
                 licence=None, url=None, default_target_attribute=None,
                 row_id_attribute=None, ignore_attribute=None,
                 version_label=None, citation=None, tag=None, visibility=None,
                 original_data_url=None, paper_url=None, update_comment=None,
                 md5_checksum=None, arff_string=None):
        # Attributes received by querying the RESTful API
        self.id = int(id) if id is not None else None
        self.name = name
        self.version = int(version)
        self.description = description
        self.format = format
        self.creator = creator
        self.contributor = contributor
        self.collection_date = collection_date
        self.upload_date = upload_date
        self.language = language
        self.licence = licence
        self.url = url
        self.default_target_attribute = default_target_attribute
        self.row_id_attribute = row_id_attribute
        self.ignore_attributes = ignore_attribute
        self.version_label = version_label
        self.citation = citation
        self.tag = tag
        self.visibility = visibility
        self.original_data_url = original_data_url
        self.paper_url = paper_url
        self.update_comment = update_comment
        self.md5_cheksum = md5_checksum
        self.arff_string = arff_string
        self.arff_data = self._get_arff()

        categorical = [False if type(type_) != list else True
                       for name, type_ in self.arff_data['attributes']]
        attribute_names = [name for name, type_ in self.arff_data['attributes']]

        if isinstance(self.arff_data['data'], tuple):
            X = self.arff_data['data']
            X_shape = (max(X[1]) + 1, max(X[2]) + 1)
            X = scipy.sparse.coo_matrix(
                (X[0], (X[1], X[2])), shape=X_shape, dtype=np.float32)
            X = X.tocsr()
        elif isinstance(self.arff_data['data'], list):
            X = np.array(self.arff_data['data'], dtype=np.float32)
        else:
            raise Exception()

        self.pickle_data = (X, categorical, attribute_names)

    def __eq__(self, other):
        if type(other) != BasicOpenMLDataset:
            return False
        elif self.id == other._id or \
                (self.name == other._name and self.version == other._version):
            return True
        else:
            return False

    def _get_arff(self):
        """Read ARFF file and return decoded arff.

        Reads the file referenced in self.data_file.

        Returns
        -------
        arff_string :
            Decoded arff.

        """

        decoder = arff.ArffDecoder()
        return decoder.decode(self.arff_string, encode_nominal=True)

    def get_data(self, target=None, target_dtype=int, include_row_id=False,
                 include_ignore_attributes=False,
                 return_categorical_indicator=False,
                 return_attribute_names=False):
        """Returns dataset content as numpy arrays / sparse matrices.

        Parameters
        ----------


        Returns
        -------

        """
        rval = []

        data, categorical, attribute_names = self.pickle_data

        to_exclude = []
        if include_row_id is False:
            if not self.row_id_attribute:
                pass
            else:
                if is_string(self.row_id_attribute):
                    to_exclude.append(self.row_id_attribute)
                else:
                    to_exclude.extend(self.row_id_attribute)

        if include_ignore_attributes is False:
            if not self.ignore_attributes:
                pass
            else:
                if is_string(self.ignore_attributes):
                    to_exclude.append(self.ignore_attributes)
                else:
                    to_exclude.extend(self.ignore_attributes)

        if len(to_exclude) > 0:
            logger.info("Going to remove the following row_id_attributes:"
                        " %s" % self.row_id_attribute)
            keep = np.array([True if column not in to_exclude else False
                             for column in attribute_names])
            data = data[:, keep]
            categorical = [cat for cat, k in zip(categorical, keep) if k]
            attribute_names = [att for att, k in
                               zip(attribute_names, keep) if k]

        if target is None:
            rval.append(data)
        else:
            if is_string(target):
                target = [target]
            targets = np.array([True if column in target else False
                                for column in attribute_names])

            try:
                x = data[:, ~targets]
                y = data[:, targets].astype(target_dtype)

                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y[:, 0]

                categorical = [cat for cat, t in
                               zip(categorical, targets) if not t]
                attribute_names = [att for att, k in
                                   zip(attribute_names, targets) if not k]
            except KeyError as e:
                import sys
                sys.stdout.flush()
                raise e

            if scipy.sparse.issparse(y):
                y = np.asarray(y.todense()).astype(target_dtype).flatten()

            rval.append(x)
            rval.append(y)

        if return_categorical_indicator:
            rval.append(categorical)
        if return_attribute_names:
            rval.append(attribute_names)

        if len(rval) == 1:
            return rval[0]
        else:
            return rval

    def _retrieve_class_labels(self):
        """Reads the datasets arff to determine the class-labels, and returns those.
        If the task has no class labels (for example a regression problem) it returns None."""

        dataAttributes = dict(self.arff_data['attributes'])
        if('class' in dataAttributes):
            return dataAttributes['class']
        elif('Class' in dataAttributes):
            return dataAttributes['Class']
        else:
            return None

    def _to_xml(self):
        """Serialize object to xml for upload

        Returns
        -------
        xml_dataset : string
            XML description of the data.
        """
        xml_dataset = ('<oml:data_set_description '
                       'xmlns:oml="http://openml.org/openml">')
        props = ['id', 'name', 'version', 'description', 'format', 'creator',
                 'contributor', 'collection_date', 'upload_date', 'language',
                 'licence', 'url', 'default_target_attribute',
                 'row_id_attribute', 'ignore_attribute', 'version_label',
                 'citation', 'tag', 'visibility', 'original_data_url',
                 'paper_url', 'update_comment', 'md5_checksum']  # , 'data_file']
        for prop in props:
            content = getattr(self, prop, None)
            if content is not None:
                xml_dataset += "<oml:{0}>{1}</oml:{0}>".format(prop, content)
        xml_dataset += "</oml:data_set_description>"
        return xml_dataset

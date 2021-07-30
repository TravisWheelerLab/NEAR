# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functionally annotate a fasta file.

Write functional predictions as a TSV with columns
- sequence_name (string)
- predicted_label (string)
- confidence (float); number between 0 and 1. An estimate of the model's
    confidence that the label is true.
- label_description (string); a human-readable label description.
"""

import decimal
import io
import logging
import os
from typing import Dict, List, Text, Tuple

from absl import app
from absl import flags
from Bio.SeqIO import FastaIO
import numpy as np
import pandas as pd
import inference
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF c++ logging set to ERROR
import tensorflow.compat.v1 as tf  # pylint: disable=g-import-not-at-top
import tqdm


_logger = logging.getLogger('proteinfer')

FLAGS = flags.FLAGS

flags.DEFINE_string('i', None, 'Input fasta file path.')
flags.DEFINE_string('o', None, 'Output write path.')

flags.DEFINE_integer(
    'num_ensemble_elements', 1,
    'In order to run with more than one ensemble element, you will need to run '
    'install_models.py --install_ensemble=true. '
    'More ensemble elements takes more time, but tends to be more accurate. '
    'Run-time scales linearly with the number of ensemble elements. '
    'Maximum value of this flag is {}.'.format(
        utils.MAX_NUM_ENSEMBLE_ELS_FOR_INFERENCE))
flags.DEFINE_float(
    'reporting_threshold',
    .5,
    'Number between 0 (exclusive) and 1 (inclusive). Predicted labels with '
    'confidence at least resporting_threshold will be included in the output.',
    lower_bound=0.,
    upper_bound=1.)

flags.DEFINE_string('model_cache_path', 'cached_models',
                    'Path from which to use downloaded models and metadata.')

# A list of inferrers that all have the same label set.
_InferrerEnsemble = List[inference.Inferrer]

# (list_of_pfam_inferrers, list_of_ec_inferrers, list_of_go_inferrers)
_Models = Tuple[_InferrerEnsemble, _InferrerEnsemble, _InferrerEnsemble]


def _num_decimal_places(f):
  """Get the number of decimal places in a float."""
  # https://stackoverflow.com/a/6190291/1445296
  return abs(decimal.Decimal('{}'.format(f)).as_tuple().exponent)


def _gcs_path_to_relative_unzipped_path(p):
  """Parses GCS path, to gets the last part, and removes .tar.gz."""
  return os.path.join(
      os.path.basename(os.path.normpath(p)).replace('.tar.gz', ''))


def _get_inferrer_paths(model_urls,
                        model_cache_path):
  """Convert list of model GCS urls to a list of locally cached paths."""
  return [
      os.path.join(model_cache_path, _gcs_path_to_relative_unzipped_path(p))
      for p in model_urls
  ]


def load_models(model_cache_path, num_ensemble_elements):
  """Load models from cache path into inferrerLists.

  Args:
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    num_ensemble_elements: number of ensemble elements of each type to load.

  Returns:
    (list_of_pfam_inferrers, list_of_ec_inferrers, list_of_go_inferrers)

  Raises:
    ValueError if the models were not found. The exception message describes
    that install_models.py needs to be rerun.
  """
  try:
    pfam_inferrer_paths = _get_inferrer_paths(utils.OSS_PFAM_ZIPPED_MODELS_URLS,
                                              model_cache_path)
    ec_inferrer_paths = _get_inferrer_paths(utils.OSS_EC_ZIPPED_MODELS_URLS,
                                            model_cache_path)
    go_inferrer_paths = _get_inferrer_paths(utils.OSS_GO_ZIPPED_MODELS_URLS,
                                            model_cache_path)

    to_return = []
    inferrer_list_paths_for_all_models = [
        pfam_inferrer_paths, ec_inferrer_paths, go_inferrer_paths
    ]
    pbar = tqdm.tqdm(
        desc='Loading models',
        position=0,
        total=len(inferrer_list_paths_for_all_models) * num_ensemble_elements,
        leave=True,
        dynamic_ncols=True)
    for inferrer_list_paths in inferrer_list_paths_for_all_models:
      inner_itr = inferrer_list_paths[:num_ensemble_elements]
      inferrer_list = []
      for p in inner_itr:
        inferrer_list.append(inference.Inferrer(p, use_tqdm=True))
        pbar.update()
      to_return.append(inferrer_list)

    pfam_inferrers = to_return[0]
    ec_inferrers = to_return[1]
    go_inferrers = to_return[2]

    return pfam_inferrers, ec_inferrers, go_inferrers

  except tf.errors.NotFoundError as exc:
    err_msg = 'Unable to find cached models in {}.'.format(model_cache_path)
    if num_ensemble_elements > 1:
      err_msg += (
          ' Make sure you have installed the entire ensemble of models by '
          'running\n    install_models.py --install_ensemble '
          '--model_cache_path={}'.format(model_cache_path))
    else:
      err_msg += (
          ' Make sure you have installed the models by running\n    '
          'install_models.py --model_cache_path={}'.format(model_cache_path))
    err_msg += '\nThen try rerunning this script.'

    raise ValueError(err_msg, exc)


def _assert_fasta_parsable(input_text):
  with io.StringIO(initial_value=input_text) as f:
    fasta_itr = FastaIO.FastaIterator(f)
    end_iteration_sentinel = object()

    # Avoid parsing the entire FASTA contents by using `next`.
    # A malformed FASTA file will have no entries in its FastaIterator.
    # This is unfortunate (instead of it throwing an error).
    if next(fasta_itr, end_iteration_sentinel) is end_iteration_sentinel:
      raise ValueError('Failed to parse any input from fasta file. '
                       'Consider checking the formatting of your fasta file. '
                       'First bit of contents from the fasta file was\n'
                       '{}'.format(input_text.splitlines()[:3]))


def parse_input_to_text(input_fasta_path):
  """Parses input fasta file.

  Args:
    input_fasta_path: path to FASTA file.

  Returns:
    Contents of file as a string.

  Raises:
    ValueError if parsing the FASTA file gives no records.
  """
  _logger.info('Parsing input from %s', input_fasta_path)
  with tf.io.gfile.GFile(input_fasta_path, 'r') as input_file:
    input_text = input_file.read()

  _assert_fasta_parsable(input_text=input_text)
  return input_text


def input_text_to_df(input_text):
  """Converts fasta contents to a df with columns sequence_name and sequence."""
  with io.StringIO(initial_value=input_text) as f:
    fasta_records = list(FastaIO.FastaIterator(f))
    fasta_df = pd.DataFrame([(f.name, str(f.seq)) for f in fasta_records],
                            columns=['sequence_name', 'sequence'])

  return fasta_df


def perform_inference(input_df, models,
                      reporting_threshold):
  """Perform inference for Pfam, EC, and GO using given models.

  Args:
    input_df: pd.DataFrame with columns sequence_name (str) and sequence (str).
    models: (list_of_pfam_inferrers, list_of_ec_inferrers,
      list_of_go_inferrers).
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.

  Returns:
    pd.DataFrame with columns sequence_name (str), label (str), confidence
    (float).
  """
  predictions = []
  for inferrer_list in tqdm.tqdm(
      models, position=1, desc='Progress', leave=True):
    predictions.append(
        inference.get_preds_at_or_above_threshold(input_df, inferrer_list,
                                                  reporting_threshold))

  print('\n')  # Because the tqdm bar is position 1, we need to print a newline.

  predictions = pd.concat(predictions)
  return predictions


def _sort_df_multiple_columns(df, key):
  """Sort df based on callable key.

  Args:
    df: pd.DataFrame.
    key: function from rows of df (namedtuples) to tuple. This is used in the
      builtin `sorted` method as the key.

  Returns:
    A sorted copy of df.
  """
  # Unpack into list to take advantage of builtin sorted function.
  # Note that pd.DataFrame.sort_values will not work because sort_values'
  # sorting function is applied to each column at a time, whereas we need to
  # consider multiple fields at once.
  df_rows_sorted = sorted(df.itertuples(index=False), key=key)
  return pd.DataFrame(df_rows_sorted, columns=df.columns)


def order_df_for_output(predictions_df):
  """Semantically group/sort predictions df for output.

  Sort order:
  Sort by query sequence name as they are in `predictions_df`.
  Put all Pfam labels first, then EC labels, then GO labels.
  Given that,
    - if it's an EC label, sort by label alphabetically.
    - else, sort by confidence descending.
      Given that, sort by description alphabetically.

  The reason to sort EC differently is that the alphabetic ordering of EC labels
  is meaningful, while the alphabetic orering of Pfam and GO labels is not.

  Args:
    predictions_df: df with columns sequence_name (str), predicted_label (str),
      confidence (float), description (str).

  Returns:
    df with columns sequence_name (str), predicted_label (str), confidence
    (float), description (str).
  """
  seq_name_to_original_order = {
      item: idx for idx, item in enumerate(predictions_df.sequence_name)
  }

  def filter_by_label_type(df, label_type):
    return df[df.predicted_label.apply(lambda x: x.startswith(label_type))]

  def _orderer_pfam_and_go(df_row):
    """See outer function doctsring."""
    confidence_sort_key = -1 * df_row.confidence
    return (seq_name_to_original_order[df_row.sequence_name],
            confidence_sort_key, df_row.description)

  def _orderer_ec(df_row):
    """See outer function doctsring."""
    confidence_sort_key = -1 * df_row.confidence
    return (seq_name_to_original_order[df_row.sequence_name],
            df_row.predicted_label)

  pfam_df = filter_by_label_type(predictions_df, 'Pfam')
  ec_df = filter_by_label_type(predictions_df, 'EC')
  go_df = filter_by_label_type(predictions_df, 'GO')

  pfam_df_sorted = _sort_df_multiple_columns(pfam_df, _orderer_pfam_and_go)
  ec_df_sorted = _sort_df_multiple_columns(ec_df, _orderer_ec)
  go_df_sorted = _sort_df_multiple_columns(go_df, _orderer_pfam_and_go)
  return pd.concat([pfam_df_sorted, ec_df_sorted, go_df_sorted])


def _format_float_confidence_for_output(input_float,
                                        num_decimal_places):
  # Create a separate function so as to test it against our expectations.
  return np.around(input_float, num_decimal_places)


def format_df_for_output(predictions_df,
                         label_to_description,
                         num_decimal_places):
  """Formats df for outputting.

  Args:
    predictions_df: df with columns sequence_name (str), predicted_label (str),
      confidence (float).
    label_to_description: contents of label_descriptions.json.gz. Map from label
      to a human-readable description.
    num_decimal_places: number of decimal places to display in the confidence
      output column.

  Returns:
    df with columns sequence_name (str), predicted_label (str), confidence
    (float), description (str).
  """
  predictions_df = predictions_df.copy()

  predictions_df['description'] = predictions_df.predicted_label.apply(
      label_to_description.__getitem__)
  predictions_df['confidence'] = predictions_df.confidence.apply(
      lambda x: _format_float_confidence_for_output(x, num_decimal_places))

  return order_df_for_output(predictions_df)


def write_output(predictions_df, output_path):
  """Write predictions_df to tsv file."""
  _logger.info('Writing output to %s', output_path)
  with tf.io.gfile.GFile(output_path, 'w') as f:
    predictions_df.to_csv(f, sep='\t', index=False)


def run(input_text, models, reporting_threshold,
        label_to_description):
  """Runs inference and returns output as a pd.DataFrame.

  Args:
    input_text: contents of a fasta file.
    models: (list_of_pfam_inferrers, list_of_ec_inferrers,
      list_of_go_inferrers).
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    label_to_description: contents of label_descriptions.json.gz. Map from label
      to a human-readable description.

  Returns:
    pd.DataFrame with columns sequence_name (str), label (str), confidence
    (float).
  """
  input_df = input_text_to_df(input_text)
  predictions_df = perform_inference(
      input_df=input_df,
      models=models,
      reporting_threshold=reporting_threshold)

  predictions_df = format_df_for_output(
      predictions_df=predictions_df,
      label_to_description=label_to_description,
      num_decimal_places=max(2, _num_decimal_places(reporting_threshold)))

  return predictions_df


def load_assets_and_run(input_fasta_path, output_path,
                        num_ensemble_elements, model_cache_path,
                        reporting_threshold):
  """Loads models/metadata, runs inference, and writes output to tsv file.

  Args:
    input_fasta_path: path to FASTA file.
    output_path: path to which to write a tsv of inference results.
    num_ensemble_elements: Number of ensemble elements to load and perform
      inference with.
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
  """
  _logger.info('Running with %d ensemble elements', num_ensemble_elements)
  input_text = parse_input_to_text(input_fasta_path)

  models = load_models(model_cache_path, num_ensemble_elements)
  label_to_description = utils.load_gz_json(
      os.path.join(model_cache_path,
                   utils.INSTALLED_LABEL_DESCRIPTION_FILE_NAME))

  predictions_df = run(input_text, models, reporting_threshold,
                       label_to_description)
  write_output(predictions_df, output_path)


def main(_):
  # TF logging is too noisy otherwise.
  tf.get_logger().setLevel(tf.logging.ERROR)

  if FLAGS.reporting_threshold == 0.:
    raise ValueError('The reporting_threshold flag was 0. Please supply a '
                     'value between 0 (exclusive) and 1 (inclusive). A value '
                     'of zero will report every label for every protein.')
  load_assets_and_run(
      input_fasta_path=FLAGS.i,
      output_path=FLAGS.o,
      num_ensemble_elements=FLAGS.num_ensemble_elements,
      model_cache_path=FLAGS.model_cache_path,
      reporting_threshold=FLAGS.reporting_threshold)


if __name__ == '__main__':
  _logger.info('Process started.')
  flags.mark_flags_as_required(['i', 'o'])

  app.run(main)

"""
Registry of experiment pipeline configs.

Each entry maps a PIPELINE_CONFIG value (e.g. "exp_001_smart_document_splitter")
to the experiment's `pipeline_config` module. The module must expose a
`pipelines` list in the same shape base pipeline_configs use
(see haystack_api/pipeline/pipeline_configs/en_gen_config.py for reference).

Adding a new experiment: create experiments/exp_NNN_<slug>/pipeline_config.py,
then add one line here.
"""

from experiments.exp_001_smart_document_splitter import pipeline_config as exp_001
from experiments.exp_002_gliner_biomedical_metadata import pipeline_config as exp_002

EXPERIMENT_CONFIGS = {
    "exp_001_smart_document_splitter": exp_001,
    "exp_002_gliner_biomedical_metadata": exp_002,
}

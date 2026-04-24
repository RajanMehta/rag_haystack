"""
Template pipeline_config for a new experiment.

Copy this folder to `experiments/exp_NNN_<slug>/`, fill in the `pipelines` list
with your Haystack pipeline specs (see
`haystack_api/pipeline/pipeline_configs/en_gen_config.py` for a full example),
and register the module in `experiments/_registry.py`.

Every component type is referenced by its fully qualified dotted path string.
For components you add under this experiment's `components/` folder, the path is:

    experiments.exp_NNN_<slug>.components.<module>.<ClassName>
"""

pipelines = [
    # {
    #     "name": "indexing_pipeline",
    #     "type": "sync",
    #     "configs": { ... Haystack Pipeline JSON ... },
    # },
    # ...
]

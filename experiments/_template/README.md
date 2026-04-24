# Experiment NNN — <name>

**PIPELINE_CONFIG value:** `exp_NNN_<slug>`

## The idea

One paragraph. What question this experiment answers and why it's interesting.

## What's new vs. base

Bulleted diff against `haystack_api/`. Which components are new, which are swapped, which schema fields are newly consumed.

## How to run

```bash
PIPELINE_CONFIG=exp_NNN_<slug> make up
```

Then one or two curl examples showing the new behavior end-to-end.

## Reading order

1. `pipeline_config.py` — the DAG
2. `components/<file>.py` — the new component(s)
3. Where it connects to base — schema fields / controllers / tasks that this experiment exercises

## Gotchas

Anything non-obvious a reader should know before reusing this code.

## Files

```
experiments/exp_NNN_<slug>/
├── README.md
├── pipeline_config.py
├── components/
│   └── <your_component>.py
└── assets/
```

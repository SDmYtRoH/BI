"""Unified RehaPiano pipeline: raw recordings -> one feature table per person -> statistics and classifier.

Stages (``scripts/run_pipeline.py``):

1. ``index``        - every ``results_*`` file: payload identity, MD5, match to metadata, game; copies collapsed;
2. ``participants`` - one row per person: cohort, questionnaire screening, age, decade, recording period;
3. ``features``     - per attempt: EXACT_FORCE finger/hand features, RHYTHM press features (incremental, cached by MD5);
4. ``aggregate``    - attempt -> session (median) -> person (mean); ``person_features.csv``;
5. ``analysis``     - hypotheses, age trends and decade norms of healthy people, classifier and ablation.

Every stage reads the previous stage's CSV files in ``outputs/pipeline/``, so a stage can be rerun alone.
New recordings or metadata files only need to be copied into ``data/``; already processed files are reused.
"""

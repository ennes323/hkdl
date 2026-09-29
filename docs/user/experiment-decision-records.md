# Experiment Decision Records

An Experiment Decision Record is an optional authored note for one meaningful
research branch. It records why the branch exists, how it relates to another
documented idea, which exact HKDL evidence was used, and what decision followed.

The record is guidance, not an execution requirement. HKDL does not discover,
validate, update, or require these files. Their presence and completeness do
not affect Experiment or Variant validation, Train, Eval, Export, status,
retry, or cleanup behavior.

## Start a record

Copy [the starter record](examples/experiment-decision-record.yaml) to:

```text
experiments/<experiment>/docs/records/<RECORD-ID>.yaml
```

Use a stable uppercase ID such as `BASE`, `M05-LR0-0085`, or
`M05-CAL-RANGE512`. The ID identifies the research branch, not its current
state or outcome. Keep exact parameter values in `lineage.changes` rather than
reconstructing them from the ID.

At creation time, record only the intent:

- a stable `id` and editable `title`;
- `kind`, `role`, and lifecycle `state`;
- the research `question`;
- one direct research parent and a typed relationship;
- the intended configuration changes;
- the owning Experiment and Variant;
- `decision.outcome: not_run` until a decision exists.

The starter is a valid root baseline example. For a child record, replace the
lineage with an explicit parent and change:

```yaml
lineage:
  parent: M05
  relation: single_axis
  changes:
    - path: train.learning_rate
      from: 0.01
      to: 0.0085
```

## Add evidence after execution

Do not duplicate metric histories, Run state, retry ancestry, or Model
manifests. Those remain authoritative HKDL records. Add only the exact evidence
selected for the research decision:

```yaml
provenance:
  experiment: demo
  variant: baseline
  runs:
    - run_id: run-001
      model_id: model-0123456789abcdef0123456789abcdef
  evaluations:
    - name: default
      run_id: run-002
```

A research parent does not prove that a checkpoint was reused. When weights or
artifacts were actually inherited, add a separate `checkpoint_parent` or
`artifact_parent` with the exact Model or artifact identity and SHA-256 digest.
Runs connected by `retry_of` remain execution lineage and do not become
research parent-child records.

## Close the decision when useful

After comparing compatible evidence, replace `not_run` and add only the fields
needed to preserve the conclusion:

```yaml
state: retired

decision:
  outcome: did_not_improve
  summary: The candidate did not improve the baseline on the same evaluation surface.
  revisit_when: Revisit with a different optimizer hypothesis.

retention:
  policy: retain_evidence
  keep:
    - decision_record
    - canonical_metrics
    - best_checkpoint_identity
```

Suggested outcomes are `selected`, `did_not_improve`, `inconclusive`,
`superseded`, `released`, and `not_run`. Suggested lifecycle states are
`active`, `deployed`, `retired`, and `deferred`.

An omitted optional field means that it does not apply yet. Use `null` only
after checking a relevant value and confirming that it is unavailable. Never
use `null` to mean "not checked."

Retention is a documentation decision, not permission to delete anything.
Any authored Variant, generated Run or Model, artifact, or dataset cleanup
remains a separate explicit operation with its own evidence and review.

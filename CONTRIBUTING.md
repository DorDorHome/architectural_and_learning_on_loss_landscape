# Contributing

Workflow
- Branch from main; one topic per PR; small, focused changes
- Include tests for learners/models/datasets when possible

Standards
- Python 3.9+, type hints when practical
- Keep factory interfaces stable; deprecate before breaking
- Document new dataset/experiment additions under `docs/`

PR checklist
- [ ] Tests added/updated (if applicable)
- [ ] Docs updated: Overview/Architecture/Extending and, if relevant, `DATA_LOADING.md` or `EXPERIMENTS.md`
- [ ] New dataset/model/learner registered in the appropriate factory
- [ ] Experiment config validated with Hydra (runs at least one epoch)

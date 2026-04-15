# Lessons

- When a workflow script is presented as an end-to-end operator command, treat missing user-space prerequisites as part of the workflow unless the user explicitly wants a manual setup boundary. For `pubmed-tei-cluster`, that means bootstrapping Rust automatically rather than stopping at "cargo not found".
- When a user asks about command behavior, answer against the specific command they named instead of broadening to adjacent commands. For CLI UX follow-ups, inspect that exact entrypoint before responding.

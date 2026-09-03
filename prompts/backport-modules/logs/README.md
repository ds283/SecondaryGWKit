# Implementation logs

One log per prompt, written by the agent that executed it and committed alongside its code changes.

Naming: `NN-<prompt-name>.md`, matching the prompt file. The template and the rules for classifying
deviations are in [`../README.md`](../README.md) §5.1.

These logs are the campaign's record of *why the shipped code differs from the plan*. They are meant
to be read instead of re-deriving that from the diff, so a deviation recorded here needs enough
context to be evaluated on its own.

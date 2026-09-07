# Logs — source remediation campaign

One log per prompt, named `NN-<name>.md` to match the prompt file, written by the agent that
executed the prompt and committed in the same commit as the code. Template and classification
rules: [`../README.md`](../README.md) §5.1.

A log is the record a later reader uses instead of re-deriving what happened from the code. It must
state what shipped, classify every deviation from the prompt as **STRUCTURALLY REQUIRED**,
**IMPLEMENTATION CHOICE** or **UNINTENDED DRIFT**, quote the numbers from every verification step,
and say what the next prompt needs to know.

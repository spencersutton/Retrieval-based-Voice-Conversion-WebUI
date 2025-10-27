---
mode: agent
model: GPT-4.1
---
Operate only on ${file} and ignore any previous context.

Simplify and make this code more readable.
Reduce duplication where possible.
Ensure there are no functional changes.
Do not remove comments unless it is commented out code 
and only add comments if they improve clarity significantly.
Replace os.*/open/etc operations with pathlib where possible.

DO NOT concern yourself with any import order warnings.
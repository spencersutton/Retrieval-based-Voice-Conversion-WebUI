from typing import Literal

# Define the allowed strings as a Literal type
PitchMethod = Literal["pm", "harvest", "crepe", "rmvpe", "fcpe"]
# Define the array of strings
PITCH_METHODS: list[PitchMethod] = ["pm", "harvest", "crepe", "rmvpe", "fcpe"]

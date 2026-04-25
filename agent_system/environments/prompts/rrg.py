RRG_SYSTEM_PROMPT = """\
You are an intelligent GUI reasoning generator. Given a task goal, an observation list, history action reasonings, \
the current GUI state, an optional GUI state after the ground-truth action, and the ground truth next action, generate a reasoning chain that leads to the next action.
# Requirements
- Your reasoning chain has 2 parts: 
\t1. Action Reasoning. A concise (1 sentence) thinking process that explains why the ground truth next action should be taken given \
the current GUI state and the accumulated observation list.
\t2. Observation Writing. Record NEW observations from the CURRENT GUI state that are useful for completing this task. \
Each observation is a concise (1 sentence) factual statement grounded in what is visible right now. Do not invent unsupported facts. Do not include coordinates or trivial details.
# Rules
- Observations are append-only in this version: each new observation is appended to the end of the bank. You MUST NOT update or remove existing observations.
- Each list element must be ONE atomic fact — do not pack multiple facts into a single string (use separate list elements instead).
- Write only facts that will help execute this task (now or in later steps); be concise, and prefer fewer observations when the screenshot alone is enough.
- Observations MUST describe the CURRENT GUI state before the next action is taken. If the after-action screenshot is provided, you may use it only to \
inform the action reasoning — NEVER to write observations.
- It is acceptable (and often correct) to return an empty list if nothing new needs to be recorded.
# Output Format
- Your output MUST be a JSON object with exactly these fields: action_reasoning (string), observation_update (list of strings, one atomic fact per element).
"""

RRG_TEMPLATE_INIT = """\
Task Goal: {task_description}
Observations: empty
History Action Reasonings: empty
Ground Truth Next Action: {ground_truth_action}
Current GUI screenshot (before the ground-truth action):
<image>{after_action_section}\
"""

RRG_TEMPLATE = """\
Task Goal: {task_description}
Observations: {observation_bank}
History Action Reasonings: {reasoning_history}
Ground Truth Next Action: {ground_truth_action}
Current GUI screenshot (before the ground-truth action):
<image>{after_action_section}\
"""

_AFTER_ACTION_SECTION = "\nAfter GUI screenshot (after the ground-truth action):\n<image>"


#set page(paper: "a4", numbering: "1")
#set heading(numbering: "1.")
#set text(hyphenate: auto)
#set par(justify: true)
#show heading: it => { it + v(0.2em) }

#title[Reverse Reasoning Generator]

= Motivation

For GUI agents, the quality of reasoning matters for their performance on *complex* tasks or *long-horizon* tasks.

Despite its importance, it is hard to get many high-quality reasonings for GUI tasks. If we ask human annotators to do so, it costs too much time and the quality is hard to control as well. In comparison, the action-only trajectories are much easier to get.

As a result, we plan to build a *Reverse Reasoning Generator* (an VLM possibly), which takes an action-only trajectory as input, and complete high-quality reasonings between actions, so that the output trajectories will help a lot for developing GUI agents. It should be noted that, the input action-only trajectory is fixed, always successful in finishing the task, and lacking of reasonings.

= Algorithm Design

We formalize reasoning generation as a *fact-augmented reverse reasoning* problem.
Because the generator works offline over successful trajectories, it may use limited hindsight information to better interpret the semantic role of the ground-truth action, while still being required to write facts only from the pre-action GUI state.
At every step, the generator must produce not only local reasoning for the current action, but also explicit memory operations:

1. *Fact Citation*: retrieve which former facts are used now;
2. *Action Reasoning*: explain why the current ground-truth action is reasonable;
3. *Fact Writing*: record new reusable observations about the *current* GUI state that may matter in future steps.

It is important to note the timing of fact writing: facts must be grounded in what the agent *currently observes* in screenshot $s_t$, before action $a_t$ is taken.
In contrast, the action-reasoning channel may additionally use hindsight signals from the successful trajectory to disambiguate why $a_t$ is the correct action.
This preserves the memory-writing semantics of a normal GUI agent, where the agent first observes the current state, writes or updates its memory, reasons about the next action, and then executes it.
Facts that predict the outcome of $a_t$ rather than describe the observable current state are incorrect by design.

This decomposition separates the two roles that were entangled in the previous design:

- local action support, which should be rewarded step by step;
- long-horizon memory construction and retrieval, which should be rewarded over the whole sampled trajectory.

== Dataset and Notation

Each training sample is a successful action-only GUI trajectory:

$
tau = (q, (s_1, a_1), (s_2, a_2), dots, (s_T, a_T))
$

where:

- $q$ is the task instruction;
- $s_t$ is the screenshot or state at step $t$;
- $a_t$ is the ground-truth action at step $t$;
- $T$ is the trajectory length.

During training, each source trajectory is rolled out $n$ times.
The generator produces one sampled reasoning trajectory per rollout.

For rollout $k$ and step $t$, the model output is:

$
y_t^(k) = (c_t^(k), r_t^(k), w_t^(k))
$

where:

- $c_t^(k)$ is the set of cited fact indices at step $t$;
- $r_t^(k)$ is the action-oriented reasoning text;
- $w_t^(k)$ is the list of newly written facts at step $t$.

Facts are stored in a trajectory-local fact bank.
Let $F_(< t)^(k)$ denote all facts written before step $t$ in rollout $k$.
Then every citation in $c_t^(k)$ must refer to an item in $F_(< t)^(k)$.

We require each written fact to be *atomic and reusable*: a fact should describe one reusable observation, status, or conclusion, instead of mixing unrelated ideas in one sentence.
One exception applies: closely related attributes of a single entity may be recorded together as a single fact, since they form a coherent unit about the same object rather than independent ideas.

== Rollout Process

For step $t$ in rollout $k$, the generator receives:

- the task $q$;
- the current screenshot $s_t$;
- an optional after-action screenshot $s_(t + 1)$ from the successful trajectory;
- the ground-truth action $a_t$;
- the current fact bank $F_(< t)^(k)$;
- the prior action-reasoning history $R_(< t)^(k)$.

Here $R_(< t)^(k)$ denotes the sequence of action reasonings generated for earlier steps in the same rollout.
This reasoning history is an auxiliary discourse channel rather than part of the explicit fact bank: it is provided to help the generator maintain coherence and learn the separation between reusable facts and transient action explanations.

The model must then generate the following three-part output:

```text
[Fact Citation]
...

[Action Reasoning]
...

[Fact Writing]
...
```

The citation part and fact-writing part may be empty.
The action-reasoning part may use $q$, $s_t$, $a_t$, the fact bank $F_(< t)^(k)$, the prior reasoning history $R_(< t)^(k)$, and optional hindsight from $s_(t + 1)$ to better infer the semantic intent of the ground-truth action.
However, all facts in $w_t^(k)$ must be grounded in the current screenshot $s_t$; they describe what is observable *before* $a_t$ is executed.
The after-action screenshot $s_(t + 1)$ must not be used as evidence for fact writing or fact citation.
After step $t$ finishes, all newly written facts in $w_t^(k)$ are appended to the fact bank and become available to later steps in the same rollout.
The newly generated action reasoning $r_t^(k)$ is appended to the reasoning history and becomes available to later steps as auxiliary context.

This rollout design has two intended properties:

- it preserves the step-local efficiency advantage over full-trajectory rollout;
- it makes memory behavior explicit, so later reward calculation can distinguish retrieval, local reasoning, and memory writing.

This rollout therefore uses *privileged hindsight for reasoning generation* but not for memory construction.
The resulting supervision is intended for training stronger GUI agents, not for exactly imitating the information constraints of a deployed online agent.

== Frozen Evaluators

We use frozen evaluators throughout training.
The generator itself is never used as its own reward model.

We assume two always-used frozen semantic evaluators:

1. *Citation judge* $J_"cite"$.
   For each step, it determines which prior facts are truly needed for the current decision.
   $J_"cite"$ receives the current screenshot $s_t$ so it can identify which prior facts contain information that is *not* directly visible on screen and is therefore worth retrieving.
   The optional after-action screenshot and prior action reasonings are not treated as citable memory items.

2. *Fact judge* $J_"fact"$.
   It checks whether a newly written fact is correct, atomic, and appropriately granular.
   $J_"fact"$ receives the current screenshot $s_t$ (the same screenshot the generator observed when writing the fact) and verifies correctness against what is actually visible at that moment.
   A fact that describes the outcome of $a_t$ rather than the observable state in $s_t$ is considered factually incorrect.

For tasks whose completion requires an explicit returned answer rather than a bare success status, we additionally use a frozen *final validator* $J_"final"$.
Given the task, the correct final answer, and the set of facts judged necessary for producing that answer, it predicts whether those facts are sufficient to derive the correct output.

These evaluators may be implemented by separate LLM judges or smaller specialized models, but all are frozen during generator training.

== Step-wise Reward: Action Reasoning

For the *Action Reasoning* part, we do not apply a semantic quality reward.
Instead, we only require the generator to follow the prescribed output format so that the three spans—fact citation, action reasoning, and fact writing—can be parsed reliably.

This choice intentionally avoids introducing a noisy semantic judge for free-form reasoning text.
As a result, the main optimization target is the model's memory behavior: which facts are retrieved and which facts are written or updated.
The additional hindsight inputs are introduced only to improve the clarity and semantic correctness of the generated action reasoning, not to redefine what counts as valid memory.

== Step-wise Reward Part 1: Fact Citation

The citation reward should encourage the model to retrieve the right prior facts when they are needed, while avoiding unnecessary citations.

For rollout $k$ and step $t$, let:

- $C_t^(k)$ be the set of cited facts;
- $N_t^(k)$ be the set of facts judged by $J_"cite"$ to be necessary for the current decision;
- $D_t^(k) = C_t^(k) backslash N_t^(k)$ be the set of over-cited facts, i.e. cited facts that were not necessary.

We then define:

$
P_t^(k) = (|C_t^(k) inter N_t^(k)|) / max(1, |C_t^(k)|)
$

$
Q_t^(k) = (|C_t^(k) inter N_t^(k)|) / max(1, |N_t^(k)|)
$

$
U_t^(k) = (|D_t^(k)|) / max(1, |C_t^(k)|)
$

where:

- $P_t^(k)$ is citation precision;
- $Q_t^(k)$ is citation recall;
- $U_t^(k)$ is the over-citation ratio.

The step-level citation score is:

$
R_"cite-step"^(k, t) =
alpha_"prec" P_t^(k) +
alpha_"rec" Q_t^(k) -
alpha_"rep" U_t^(k)
$

If both $C_t^(k)$ and $N_t^(k)$ are empty, we assign a special null-citation reward:

$
R_"cite-step"^(k, t) = alpha_"prec" + alpha_"rec"
$

This term rewards the model for correctly citing nothing when no prior memory is needed.

This design directly models the three desired properties:

- cite relevant facts;
- avoid missing necessary facts;
- avoid over-citation.

== Step-wise Reward Part 2: Fact Writing

The fact-writing reward should encourage the model to write facts that are:

- correct;
- atomic and non-overlapping;
- not overly detailed or fragmented;
- actually useful for future reasoning.

Unlike the append-only formulation, the fact bank here is *versioned*.
At each step, the generator may either add a new fact or update an existing fact.
For an observation slot $i$, let

$
f_(i, 0), f_(i, 1), dots, f_(i, L_i)
$

denote the sequence of fact versions written to that slot in chronological order.
Each version has a write step $t_(i, j)$ and content judged against the screenshot observed at that step.

For a fact version $f_(i, j)$, the fact judge $J_"fact"$ provides:

- $V(f_(i, j)) in [0, 1]$: factual correctness;
- $A(f_(i, j)) in [0, 1]$: atomicity / logical independence;
- $G(f_(i, j)) in [0, 1]$: granularity quality, where lower quality means either too detailed or too vague.

We additionally define a *future validated use* term.
At a later step $u$, if fact index $i$ is cited and judged necessary, the credited version is the latest version written strictly before step $u$.

For a specific version $f_(i, j)$, let

$
cal(U)(f_(i, j)) = {u > t_(i, j) : f_(i, j) "is active at step" u}
$

where "active at step $u$" means that $f_(i, j)$ is the latest version of fact index $i$ written strictly before step $u$, and the citation of index $i$ at step $u$ is judged necessary.

For each future use $u in cal(U)(f_(i, j))$, let $m_(f_(i, j), u)$ be the number of earlier validated uses of the same version before step $u$.
Then the discounted utility of fact version $f_(i, j)$ is:

$
T(f_(i, j)) = sum_(u in cal(U)(f_(i, j))) gamma^((u - t_(i, j) - 1) / max(1, T - 1)) / (1 + m_(f_(i, j), u))
$

This trajectory-relative discounting is intentional.
It prevents a fact whose value appears only near the end of a long-horizon trajectory from being assigned almost zero reward.
In particular, it preserves credit for intermediate results that may be irrelevant for most steps but crucial for a late decision.
As before, the first validated use receives the highest weight, and repeated later uses are down-weighted by $(1 + m)^(-1)$.

Define the intrinsic quality score of a fact version as:

$
Q(f_(i, j)) =
V(f_(i, j)) (lambda_"val" +
lambda_"atom" A(f_(i, j)) +
lambda_"gran" G(f_(i, j)))
$

Correctness $V$ multiplies the entire expression so that a factually incorrect observation ($V = 0$) receives zero quality regardless of its structural quality.
A fact that is well-formed but describes a state that does not exist in the current screenshot should not be rewarded at all.

If $cal(U)(f_(i, j)) = emptyset$, we assign

$
R_"fact"(f_(i, j)) = 0
$

That is, a fact version receives no reward unless it has at least one validated future use.
This avoids rewarding facts that are locally well-formed but never prove useful later in the trajectory.

Otherwise, for an *add* operation, the fact-version reward is:

$
R_"fact"(f_(i, j)) = lambda_"use" T(f_(i, j)) + Q(f_(i, j))
$

For an *update* operation, let the previous version in the same slot be $f_(i, j - 1)$.
We define the positive revision gain as

$
Delta(f_(i, j)) = max(0, Q(f_(i, j)) - Q(f_(i, j - 1)))
$

and reward the update by

$
R_"fact"(f_(i, j)) = lambda_"use" T(f_(i, j)) + eta_"upd" Delta(f_(i, j))
$

where $eta_"upd"$ is an update bonus scale.
Thus, an update is rewarded for future usefulness plus positive quality improvement, rather than receiving the full quality term again.

For step $t$, we collect all fact-version rewards whose write step is $t$ and define the step-wise writing reward

$
R_"write-step"^(k, t) = sum_(f : t_f = t) R_"fact"(f)
$

Thus, future usefulness is still evaluated over the whole trajectory, but the resulting reward is assigned back to the step that wrote the fact.

== Trajectory-wise Reward: Final Validation

For tasks that terminate with a bare finish signal and no explicit returned answer, we define

$
R_"final"^k = 1
$

since there is no additional answer string to validate.

For tasks with an explicit returned answer, we construct a final-answer fact set from the facts judged necessary for producing that answer and ask $J_"final"$ whether those necessary facts are sufficient to derive the correct output.
The final validation reward is binary:

$
R_"final"^k in {0, 1}
$

where $R_"final"^k = 1$ if the necessary facts are sufficient to support the correct answer, and $R_"final"^k = 0$ otherwise.

This trajectory-level term captures whether the overall memory process preserves enough information to support the final task outcome.

For each source trajectory, the $n$ sampled rollouts form one *trajectory group*.
Group-based normalization is then applied within that group to obtain:

$
A_"final"^k
$

== Final Token-level Training Signal

The final PPO / GRPO training signal is assigned by output span.
Since no semantic step-level reasoning reward is used, only the memory-related spans receive learned reward.

For each source trajectory step $t$, the $n$ sampled rollouts also form a *step group* for that step.
Group-based normalization is applied within each step group to obtain:

$
A_"cite"^(k, t)
$

from $R_"cite-step"^(k, t)$, and

$
A_"write"^(k, t)
$

from $R_"write-step"^(k, t)$.

For a token generated at step $t$ in rollout $k$:

- if the token belongs to the *Fact Citation* span, it receives
  $
  A = w_"cite-step" A_"cite"^(k, t) + w_"final" A_"final"^k
  $
- if the token belongs to the *Fact Writing* span, it receives
  $
  A = w_"write-step" A_"write"^(k, t) + w_"final" A_"final"^k
  $
- if the token belongs to the *Action Reasoning* span, it receives no semantic reward beyond any external format-validity constraint used by the implementation.

This masked credit assignment matches the intended role decomposition:

- citation quality is optimized by step-local comparison across rollouts;
- fact writing is evaluated by delayed future use but credited back to the responsible step;
- final-answer sufficiency is optimized by a binary trajectory-level signal.

== Summary of the Intended Behavior

Under this design, the generator is encouraged to learn three coordinated behaviors:

1. retrieve old facts only when they are actually needed;
2. maintain the required structured output format for action reasoning;
3. write compact reusable facts that later become genuinely useful and collectively preserve enough information for the final outcome.

Compared with the previous fold/unfold memory reward, this fact-based design gives a more direct training signal for delayed-use memory and long-horizon information tracking.

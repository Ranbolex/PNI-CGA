# Algorithm and implementation

## State and evidence

`AcquisitionModel` holds a finite set of **joint** configurations `h`, each with
a persistent ID, answer `y(h)`, facts, optional anonymous status and positive
mass. Mass affects answer-balanced tie-breaking, never membership or exclusion.
Entity identity, shared relation endpoints and uninspected count residuals
should be represented in joint facts/configurations rather than independent
regional answer labels.

The frontier `H` is the retained set of IDs. Exact same-answer compression
stores the union of members. It does not replace the union with a single
representative or discard a low-mass alternative. This finite representation
is an exact denotation; a symbolic task adapter can provide conservative
overapproximations before producing this interface.

For an acquired action `a` and verified outcome `o`, updates are:

```text
U(H, a, o) = H intersection compatibility[a][o]
```

`compatibility[a][UNKNOWN]` must contain every configuration. The prediction
table `channels[a][h][o]` is separate: a zero probability never authorizes an
exclusion. The model can represent false exclusions; such outcomes receive
zero closure credit when they delete the true configuration.

## Closure game

For one committed tree `pi_k` and one fixed configuration `h`:

```text
J[k,h] = P_h[ h is retained at the leaf AND
              every retained configuration has answer y(h) ]

maximize gamma over x >= 0, sum(x) = 1
subject to sum_k x[k] * J[k,h] >= gamma for EVERY h
```

`evaluate_policy` takes the trajectory-level conjunction **before** the
expectation. The latent configuration is fixed throughout a tree. Outcomes
of distinct action IDs are conditionally independent given that configuration.
Encode persistent correlations as additional joint configurations. Repeated
action IDs on one path are rejected; a distinct scale/test needs a distinct
action with an appropriate joint channel.

Each branch respects the hard view/output-token/call/time budget and maximum
depth. STOP is always feasible. Empty frontiers are conflicts, not success.
A nonempty single-answer frontier is a model-relative determination.

`solve_mixture` solves three LPs in order:

1. Maximize worst-configuration closure.
2. Preserve that value and maximize answer-balanced mean closure.
3. Preserve both and minimize answer-balanced predicted cost.

For answer class `Y`, `mu(h) = mass(h) / (number_of_answers * sum_mass(Y))`.
With equal mass within equally sized answer classes, this is uniform.
LP preservation tolerances are `1e-10`; comparisons use numerical tolerances.
The supplied rational certificate provides exact arithmetic for its finite
analytic construction.

## Counterexample-guided expansion

`JointFrontier.counterexample` finds retained `h,g` with different answers and
reports differing fact keys. During best-response search this witness orders
candidate reads by predicted ability to exclude its opposite member. Actions
are ranked, rather than greedily accepted: recursion evaluates complete
outcome-contingent plans, including complementary reads with no immediate
answer payoff. All candidate leaves use the full retained frontier.

`CGAPlanner` starts with STOP, solves the restricted policy game, obtains dual
world weights and searches for a shared-policy best response. A new tree is
added and the game is solved again. When the primary objective is settled,
reduced-cost oracles price the mean and cost LPs on their respective optimal
faces. This matters when an uncloseable configuration forces the maximin
value to zero: the planner still obtains useful evidence in other worlds.

The action beam, node budget, depth, refinement rounds and wall-clock limit
bound the search. A pruned response is a feasible candidate, not an exhaustive
best-response certificate. The public `best_response` API can also optimize
weighted closure minus independently weighted predicted cost.

## Bounds and search status

- `lower_bound`: minimum per-configuration closure of the returned mixture,
  evaluated against every input configuration. Restricting the policy library
  produces a lower bound on the optimum over the full policy class.
- `upper_bound`: initially one; tightened only by a complete primary
  best response against a feasible dual distribution. A restricted library's
  optimum alone is not a full-model upper bound.
- `oracle_complete`: at least one exhaustive primary best-response query
  supplied a full-class bound during this search.
- `tie_breaks_complete`: primary, mean and cost pricing all finished without
  an improving missing policy. A small primary gap alone does not establish
  globally optimal tie-breaking.
- `certified_gap`: upper minus lower, subject to floating-point LP tolerance.

`solve_exact` enumerates the finite no-repeat class, including early STOP,
and solves all three LPs. Its `max_trees` cap raises an error rather than
returning an incomplete enumeration marked exact. Small models are intended
for this route; contingent tree counts grow combinatorially.

Bounds describe worst-configuration **closure**. For the `edge` control,
`optimization_objective` identifies its different primary objective;
`worst_residual` reports the optimized residual mass. Its achieved closure is
a lower bound, while the global closure upper bound remains one. Exhaustive
edge optimization does not certify that its closure is globally maximal.

## Execution

`run_episode` samples one tree from the returned mixture with the requested
seed and commits to it. Actual verified outcomes select child branches.
Prediction-only outcomes never update the actual frontier. The planner has
no true-world argument; the verifier receives only the delivered crop, the
literal primitive and anchor metadata.

Execution returns raw views separately for a downstream answerer. It stops
on determination, conflict, exhausted resources, an explicit terminal tree or
tool failure. A task adapter that discovers new entities can create a logged
model revision and a new plan; already spent resources and verified evidence
must be carried forward by that adapter.

## Matched residual-pair control

The included `edge` solver minimizes worst-world expected remaining
opposite-answer pair mass over the same policy class. It then maximizes
worst closure on the edge-optimal face, mean closure and finally cost. This
is an objective comparison with matched states, channels, hard budgets and
randomization. The control isolates expected residual pair mass as the
alternative to trajectory-level complete closure.

## Verification map

| Behavior | Tests / artifact |
|---|---|
| Complete closure vs residual objective | `test_manuscript_separation_under_identical_policy_class` |
| Independent exact construction | `fixtures/separation_certificate.json`, `verify-certificate` |
| Zero maximin still distinguishes mean | `test_zero_worst_case_still_optimizes_mean_closure` |
| Lexicographic pricing vs exhaustive reference | `test_lexicographic_column_generation_matches_full_enumeration` |
| Complementary noisy observations | `test_complementary_reads_score_joint_success_before_expectation` |
| Predictive scores cannot delete worlds | `test_predictive_zero_does_not_authorize_exclusion` |
| False exclusion cannot earn closure | `test_false_exclusion_of_truth_earns_zero_even_at_single_answer_leaf` |
| Committed mixture and all resource dimensions | `tests/test_planning.py`, `tests/test_execution.py` |
| Crop gates and provider metering | `tests/test_execution.py`, `tests/test_model_client.py` |

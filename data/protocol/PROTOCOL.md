# The frozen protocol

This is the protocol every record in the package was produced under. It is
stated here so the records can be read without the project's working notes.

## Device and operating point

Three-channel silicon-on-insulator WDM multiplexer; an $80 \times 100$ binary
pixel region, one input port and three output monitors P1--P3; 1550 nm, port P1.
An **anchor** is one random initialization of this single topology. Anchors
differ in landscape state, not in device.

## Gate order

```
proposal  ->  20-evaluation probe  ->  gate-loss routing  ->  main run
```

The order is fixed before any run. The proposal rule generates candidate flips
or patches; the verifier decides which evaluated candidates enter the
trajectory.

## The probe

A warm-up (at most 50 evaluations, 5 accepts) followed by **20 patch
evaluations**, round-robin over four adjoint-ranking tiers with spatial
interleaving. The tiers start at adjoint rank 51, so the globally top-ranked
candidates are never evaluated and the measured loss describes the pool from
rank 51 downward.

**Every evaluated candidate is recorded, accepted or not.** This is what makes
the records verifier-agnostic: the same probe can be re-read under either
acceptance rule, which is how `gate_loss` is computed.

## The two acceptance rules

| Rule | Accepts when |
|---|---|
| null gate | `dJ > T_null(n) = mu*n + z*gamma*sigma*sqrt(n)`, with `z = 2` |
| sign gate | `dJ > 0` |

`mu` and `sigma` are measured from single-pixel null draws; `gamma` calibrates
the `sqrt(n)` scaling from batch-5 null draws. The null gate is the extremal
member of the class of rules that require an action to beat a floor rather than
zero; it is the instrument here, not a recommended default.

## The measured quantity

```
gate_loss_mass = 1 - Y_gate / Y_pos
```

where `Y_pos` is the latent positive yield of the probe pool and `Y_gate` the
yield the rule actually extracts. Mass governs; the count analogue
(`1 - n_accept / n_pos`) is reported alongside it.

## Endpoint

The endpoint is the **independently simulated fresh gain**: the device is
re-simulated after the run, so the endpoint does not depend on the acceptance
rule's own bookkeeping. The trajectory's accumulated gain (the *online* value)
is not the endpoint, and the two differ by tens of percent in either direction.
Every headline number in the paper is a fresh value.

## Budgets

Per anchor: 200 evaluations for the main run; the probe is 20 of them, about
10 percent. The probe budget is a declared choice, not a measurement: repeating
it at 40, 60 and 80 evaluations on the same anchors under the same random seed
is reported in the paper, and the cost of settling a label is anchor-specific.

## What is deliberately not claimed

* The acceptance rule is changed at **both** the warm-up and the execution
  stage, so a comparison between rules is a protocol-level contrast, not an
  isolated execution-stage verifier effect.
* No fixed verifier and scheduler combination is claimed to be best across
  anchors, and the routing rule is reported as a decision input derived from the
  measurement rather than as an optimal policy.
* The mechanism is a conditional statement about the landscape state, not a
  claim about a device class.

"""
Budget tracker: every FDTD call is charged here
"""

import json
from dataclasses import dataclass, field


@dataclass
class BudgetLedger:
    """Record each FDTD charge."""
    total_budget: int
    eval_count: int = 0
    entries: list = field(default_factory=list)

    # per-category counters
    null_k1_count: int = 0
    null_k5_count: int = 0
    null_matched_count: int = 0
    probe_count: int = 0
    candidate_accept_count: int = 0
    candidate_reject_count: int = 0
    candidate_test_count: int = 0
    patch_accept_count: int = 0
    patch_reject_count: int = 0
    grad_refresh_count: int = 0  # reported separately, not charged
    grad_refresh_wall_clock: float = 0.0
    fresh_sim_count: int = 0  # reported separately

    def remaining(self) -> int:
        return self.total_budget - self.eval_count

    def is_exhausted(self) -> bool:
        return self.eval_count >= self.total_budget

    def charge(self, category: str, n: int = 1, wall_clock_s: float = 0.0):
        """Record a charge. Only endpoint-eval categories count against the budget."""
        entry = {
            "category": category,
            "count": n,
            "wall_clock_s": wall_clock_s,
            "eval_remaining": self.total_budget - self.eval_count - (
                n if category != "grad_refresh" and category != "fresh_sim" else 0
            ),
        }

        if category == "null_k1":
            self.null_k1_count += n
            self.eval_count += n
        elif category == "null_k5":
            self.null_k5_count += n
            self.eval_count += n
        elif category == "null_matched":
            self.null_matched_count += n
            self.eval_count += n
        elif category == "probe":
            self.probe_count += n
            self.eval_count += n
        elif category == "candidate_accept":
            self.candidate_accept_count += n
            self.eval_count += n
        elif category == "candidate_reject":
            self.candidate_reject_count += n
            self.eval_count += n
        elif category == "candidate_test":
            self.candidate_test_count += n
            self.eval_count += n
        elif category == "patch_accept":
            self.patch_accept_count += n
            self.eval_count += n
        elif category == "patch_reject":
            self.patch_reject_count += n
            self.eval_count += n
        elif category == "grad_refresh":
            self.grad_refresh_count += n
            self.grad_refresh_wall_clock += wall_clock_s
            # NOT counted in eval_count
        elif category == "fresh_sim":
            self.fresh_sim_count += n
            # NOT counted in eval_count

        self.entries.append(entry)

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "eval_spent": self.eval_count,
            "eval_remaining": self.remaining(),
            "breakdown": {
                "null_k1": self.null_k1_count,
                "null_k5": self.null_k5_count,
                "null_matched": self.null_matched_count,
                "probe": self.probe_count,
                "candidate_accept": self.candidate_accept_count,
                "candidate_reject": self.candidate_reject_count,
                "candidate_test": self.candidate_test_count,
                "patch_accept": self.patch_accept_count,
                "patch_reject": self.patch_reject_count,
            },
            "not_counted": {
                "grad_refresh": self.grad_refresh_count,
                "grad_refresh_wall_clock_s": self.grad_refresh_wall_clock,
                "fresh_sim": self.fresh_sim_count,
            },
        }

    def to_dict(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "eval_count": self.eval_count,
            "entries": self.entries,
            "null_k1_count": self.null_k1_count,
            "null_k5_count": self.null_k5_count,
            "null_matched_count": self.null_matched_count,
            "probe_count": self.probe_count,
            "candidate_accept_count": self.candidate_accept_count,
            "candidate_reject_count": self.candidate_reject_count,
            "candidate_test_count": self.candidate_test_count,
            "patch_accept_count": self.patch_accept_count,
            "patch_reject_count": self.patch_reject_count,
            "grad_refresh_count": self.grad_refresh_count,
            "grad_refresh_wall_clock": self.grad_refresh_wall_clock,
            "fresh_sim_count": self.fresh_sim_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetLedger":
        obj = cls(total_budget=d["total_budget"])
        obj.eval_count = d["eval_count"]
        obj.entries = d.get("entries", [])
        obj.null_k1_count = d.get("null_k1_count", 0)
        obj.null_k5_count = d.get("null_k5_count", 0)
        obj.null_matched_count = d.get("null_matched_count", 0)
        obj.probe_count = d.get("probe_count", 0)
        obj.candidate_accept_count = d.get("candidate_accept_count", 0)
        obj.candidate_reject_count = d.get("candidate_reject_count", 0)
        obj.candidate_test_count = d.get("candidate_test_count", 0)
        obj.patch_accept_count = d.get("patch_accept_count", 0)
        obj.patch_reject_count = d.get("patch_reject_count", 0)
        obj.grad_refresh_count = d.get("grad_refresh_count", 0)
        obj.grad_refresh_wall_clock = d.get("grad_refresh_wall_clock", 0.0)
        obj.fresh_sim_count = d.get("fresh_sim_count", 0)
        return obj

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "summary": self.summary(),
                "entries": self.entries,
            }, f, indent=2)


class BudgetExhausted(Exception):
    """Raised when the budget is exhausted."""
    pass


def check_budget(ledger: BudgetLedger, needed: int = 1):
    """Check the budget and raise if it cannot cover the request."""
    if ledger.eval_count + needed > ledger.total_budget:
        raise BudgetExhausted(
            f"Budget exhausted: {ledger.eval_count}/{ledger.total_budget} "
            f"(needed {needed})"
        )

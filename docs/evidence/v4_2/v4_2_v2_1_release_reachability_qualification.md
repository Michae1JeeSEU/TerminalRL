# RELEASE-family reachability qualification

| Family | Independent witnesses | Ledger entries | Exact replay tapes | Status |
| --- | ---: | ---: | ---: | --- |
| `clear_outer_RELEASE` | 10 | 240 | 10 | STABLY_REACHABLE |
| `middle_interface_RELEASE` | 10 | 600 | 10 | STABLY_REACHABLE |
| `post_WAIT_RELEASE` | 10 | 5,040 | 10 | STABLY_REACHABLE |
| `post_conflict_RELEASE` | 10 | 5,040 | 10 | STABLY_REACHABLE |

The dynamic transition recipes establish a same-UAV, same-lifecycle WAIT-to-RELEASE transition. `post_conflict` additionally preserves a preceding `geometry_conflict` WAIT and verifies that current RELEASE occurs after the blocking interface predicate has cleared.

This supersedes the former source-name classification failure: all four target families are reachable under validated legal recipes. It does **not** change historical I0 failure results or authorize formal recollection yet, because the pre-registered Teacher-selected diversity gate remains unresolved.

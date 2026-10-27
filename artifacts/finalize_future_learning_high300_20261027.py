"""Apply the supplied post-archive primary review to derived high300 reports only."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

MARKER="<!-- HIGH300_PRIMARY_REVIEW_20261027 -->"
QA="PRIMARY_VIEWED_WITH_LAYOUT_LIMITATION"
IMAGE_SHA="94b419b6f6235418ff61c015b8f99fb1917f2572b9957023fdfa687e0151a488"
CONCLUSION="One high-traffic 300 s pair: L lands one additional normal aircraft (15 vs 14), but total normal planning work and dispatch work increase by 3.52% and 9.12%; no overall online acceleration or stable quality advantage is established."
LIMITATION="Time and landing trends are readable; the right-side status labels for the final tasks overlap. This is a layout limitation and the figure is not publication-ready."

# This interpretation was supplied by the primary reviewer, not selected by
# automatic metric thresholds. It authorizes no algorithm change or new run.
REVIEW_TEXT="""## Primary review: completed high-traffic 300 s pair

One frozen-model pair completed and archived with exit code zero. L has one more normal landing (15 versus H14), zero emergency landings and zero failures. All 14/15 normal landings are from the initial 36 airborne aircraft; all 13 new requests remain airborne. Both arms admitted 49 aircraft, with H35/L34 pending. This is a descriptive single-pair result without a significance, generalization, deployment or isolated training-update causal claim.

| Saved measure | H | L |
| --- | ---: | ---: |
| All-task normal planning work (s) | 189.537665689 | 196.209438064 |
| Dispatch work (s) | 208.654763247 | 227.677272771 |
| Finished-task P95 (s; sample count) | 15.9125 (13) | 15.8024 (14) |
| Joined / finished nonempty groups | 8 / 11 | 8 / 11 |
| Pending nonempty groups | 1 | 1 |
| Submitted plans | 58 | 53 |
| Terminal energy (J) | 849682830.4738 | 860493251.5005 |
| Holding (aircraft s) | 7754.51167 | 8040.67272 |
| Physical attempts | 829 | 707 |
| CP solve (s) | 24.9756 | 17.8549 |
| Trajectory validation (s) | 53.4995 | 58.5525 |
| Prefix validation (s) | 13.8051 | 18.0888 |

L's all-task normal planning work is 3.52% higher and dispatch work 9.12% higher; its slightly lower finished-task P95 uses a different number of completed samples and does not establish overall online acceleration. Energy is 1.27% higher with 3.69% more holding. Normal-landing counts at 60/120/180/240/300 s are H 0/6/8/11/14 and L 0/6/6/11/15: the lead changes, without a stable quality advantage. The normal-lifecycle all-task work is the planning measure; full_online_wall_s also includes setup/checkpoint I/O and is not pure planning.

Fewer attempts and less CP work do not establish efficient candidate identification. L spends 4.01086 s on features plus 18.79356 s on inference (22.8044 s), performs 499 rankings with zero cache hits, and has more trajectory and prefix validation work. Its two scheduler fallbacks are NO_SOLVER_TIME/HINT_BUDGET, not model fallbacks; the model fallback map is empty.

Each arm has two READY_DEADLINE_MISSED rejections. H also has one ENERGY_BOUND_CHANGED and L one TRAFFIC_CHANGED_AFTER_DEADLINE; neither has packet expiry. One late H task and two late L tasks already have search certificates but incur about H9.90/L12.01 and L8.68 s of online-service delay before preparation, so this cannot be attributed solely to CP or the model. Revalidation is H4 rounds (3 certified, 1 late), about 0.1321 s, and L2 certified rounds, about 0.1590 s; it is not a dominant total-time term.

The shared emergency service measures handle_emergencies calls, including rescue-opportunity search/checking for low-SOC or warning/eligible aircraft even if required=False is later returned. Its H10.6117/L21.4542 s is shared rescue search/checking overhead, not an executed emergency-landing count. Neither arm declared, triggered or left unresolved an emergency; final rescue_warning counts are H1/L3 state counts, not cumulative events. The 10.84 s service difference is observed under different physical states and does not isolate a model effect. Emergency rules were not changed.

Saved replay validation, file hashes and P/V/A checks pass (H16906/L16362 samples), with H58/L53 C2 checks and zero start-time mismatches. These saved checks do not replace human replay acceptance, which remains HUMAN_PENDING.

The primary viewed planning_timeline.png: numeric/time/landing trends are readable, but right-side final-task status labels overlap. Figure review is PRIMARY_VIEWED_WITH_LAYOUT_LIMITATION, not publication-ready. No redraw or additional experiment was performed. The model, 20/5/1/4 timing, CPU12 and budgets stayed frozen; no retraining or tuning occurred.
"""


def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()
def replace_bytes(path,value):
    path=Path(path)
    if path.exists() and path.read_bytes()==value:return
    partial=path.with_name(path.name+".primary.partial")
    with partial.open("wb") as stream:stream.write(value)
    os.replace(partial,path)
def replace_json(path,value):replace_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())


def finalize(output):
    output=Path(output).resolve();analysis=load(output/"analysis_summary.json")
    protocol=analysis["scope"]["protocol"]
    if (protocol.get("root_id"),protocol.get("target_sim_seconds"))!=("s3r24",300):raise ValueError("expected completed s3r24 high300 report")
    if any(analysis["cells"][a].get("completed_boundary_s")!=300 for a in ("heuristic_reduced","learned_reduced")):raise ValueError("both 300 s results required")
    if file_hash(output/"planning_timeline.png")!=IMAGE_SHA:raise ValueError("primary-viewed timeline image differs")
    review_path=output/"primary_review.json"
    previous=load(review_path) if review_path.is_file() else {}
    review=dict(schema="future_learning_high300_primary_review_v1",review_stage="POST_ARCHIVE_PRIMARY_REVIEW",
        primary_review="PRIMARY_REVIEWED_SINGLE_PAIR_NO_OVERALL_ACCELERATION",conclusion=CONCLUSION,
        visual_review=QA,layout_limitation=LIMITATION,publication_ready=False,
        image_path="planning_timeline.png",image_sha256=IMAGE_SHA,human_replay_acceptance="HUMAN_PENDING",
        evidence_type="newly_run",generalization_claim=False,deployment_claim=False,isolated_model_causal_claim=False,
        no_more_fit_or_eval=True,model_sha256=analysis["model_sha256"],source_hash=analysis["scope"]["source_hash"],
        helper_sha256=file_hash(Path(__file__)),archive_receipt_preserved=True,
        previous_scientific_interpretation=previous.get("previous_scientific_interpretation",analysis.get("scientific_interpretation")),
        previous_figure_qa=previous.get("previous_figure_qa",analysis.get("figure_qa")))
    report_path=output/"FINAL_REPORT.md";original=report_path.read_bytes()
    backup=output/"FINAL_REPORT.before_primary_review.md"
    if not backup.exists():
        with backup.open("xb") as stream:stream.write(original)
    text=original.decode("utf-8-sig").split(MARKER,1)[0].rstrip()
    text=text.replace("Final interpretation: PENDING_PRIMARY_REVIEW. Figure review and replay human acceptance remain pending.",
        "Final interpretation: "+CONCLUSION+" Figure review: "+QA+". "+LIMITATION+" Replay human acceptance remains HUMAN_PENDING.")
    text=text.replace("causal attribution remains PENDING_PRIMARY_REVIEW.","causal attribution is not established by this single pair.")
    text=text.replace("Final image visual QA: PENDING_PRIMARY_VIEW_IMAGE. Replay HUMAN_PENDING is unchanged.",
        "Final image visual QA: "+QA+". "+LIMITATION+" Replay HUMAN_PENDING is unchanged.")
    if "PENDING_PRIMARY_REVIEW" in text or "PENDING_PRIMARY_VIEW_IMAGE" in text:raise ValueError("unrecognized generated pending-review text; retain source reports")
    replace_bytes(report_path,(text+"\n\n"+MARKER+"\n\n"+REVIEW_TEXT).encode())
    readme_path=output/"README.md";readme=readme_path.read_bytes()
    readme_backup=output/"README.before_primary_review.md"
    if not readme_backup.exists():
        with readme_backup.open("xb") as stream:stream.write(readme)
    readme_text=readme.decode("utf-8-sig").split(MARKER,1)[0].rstrip()
    replace_bytes(readme_path,(readme_text+"\n\n"+MARKER+"\n\nPrimary review complete: "+CONCLUSION+"\n\nFigure review: "+QA+". "+LIMITATION+" Replay remains HUMAN_PENDING. See FINAL_REPORT.md and primary_review.json.\n").encode())
    qa=load(output/"figure_QA.json");qa.update(visual_review=QA,layout_limitation=LIMITATION,publication_ready=False,
        image_sha256=IMAGE_SHA,human_replay_acceptance="HUMAN_PENDING")
    analysis.update(scientific_interpretation=CONCLUSION,figure_qa=qa,primary_review=review,human_replay_acceptance="HUMAN_PENDING")
    replace_json(output/"figure_QA.json",qa);replace_json(output/"analysis_summary.json",analysis);replace_json(review_path,review)
    return dict(status="PRIMARY_REVIEW_APPLIED",visual_review=QA,human_replay_acceptance="HUMAN_PENDING",output=str(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv);print(json.dumps(finalize(args.output),sort_keys=True))


if __name__=="__main__":main()

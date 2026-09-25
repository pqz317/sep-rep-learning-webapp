"""Summarize participant demographics (age range/median, self-reported sex) from the
Prolific export <data-dir>/demographic.csv.

Participants are filtered the same way as analyze_data.py (users with no prolific ID
or missing survey tags are dropped, plus --exclude-test / --exclude-bad-users), then
matched to the export by prolific ID.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_data as ad
from constants import resolve_data_dir

DEMOGRAPHIC_FILENAME = "demographic.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="resub2",
        help="Dataset name or path to a data directory containing demographic.csv "
             "(default: prolific_data/resub2/).",
    )
    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude participants whose prolific ID contains 'test'.",
    )
    ad.add_exclude_bad_users_arg(parser)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")

    user_ids = ad.load_user_ids(data_dir)
    prolific_ids = ad.load_prolific_ids(data_dir)
    exclude_ids = ad.load_excluded_user_ids(data_dir, exclude_test=args.exclude_test,
                                            exclude_bad_users=args.exclude_bad_users)
    kept = [uid for uid in user_ids if uid not in exclude_ids]
    print(f"Users: {len(user_ids)} loaded, {len(exclude_ids)} excluded, {len(kept)} kept")

    kept_pids = {uid: prolific_ids.get(uid, "").strip() for uid in kept}
    no_pid = sorted(uid for uid, pid in kept_pids.items() if not pid)
    if no_pid:
        print(f"WARNING: {len(no_pid)} kept user(s) have no prolific ID and can't be "
              f"matched to demographics: {no_pid}")
    pid_counts = pd.Series([pid for pid in kept_pids.values() if pid]).value_counts()
    for pid in pid_counts[pid_counts > 1].index:
        uids = sorted(uid for uid, p in kept_pids.items() if p == pid)
        print(f"WARNING: prolific ID {pid} appears under multiple kept users {uids}; "
              f"counted once")

    demo = pd.read_csv(os.path.join(data_dir, DEMOGRAPHIC_FILENAME), dtype=str)
    demo["Participant id"] = demo["Participant id"].str.strip()
    demo = demo[demo["Participant id"].isin({pid for pid in kept_pids.values() if pid})]
    demo = demo.drop_duplicates("Participant id")

    unmatched = sorted(uid for uid, pid in kept_pids.items()
                       if pid and pid not in set(demo["Participant id"]))
    if unmatched:
        print(f"WARNING: {len(unmatched)} kept user(s) not found in {DEMOGRAPHIC_FILENAME}: "
              f"{[(uid, kept_pids[uid]) for uid in unmatched]}")

    print(f"\nParticipants with demographics: {len(demo)}")

    # Prolific reports withheld values as e.g. CONSENT_REVOKED / DATA_EXPIRED.
    age = pd.to_numeric(demo["Age"], errors="coerce").dropna()
    print(f"\nAge (n={len(age)}): range {age.min():.0f}-{age.max():.0f}, "
          f"median {age.median():g}")
    if len(age) < len(demo):
        print(f"  {len(demo) - len(age)} participant(s) with no numeric age")

    # "Time taken" is in seconds.
    minutes = pd.to_numeric(demo["Time taken"], errors="coerce").dropna() / 60
    print(f"\nTime taken (n={len(minutes)}): mean {minutes.mean():.1f} min, "
          f"median {minutes.median():.1f} min")

    sex = demo["Sex"].fillna("Not reported")
    print("\nSelf-reported sex:")
    for label, count in sex.value_counts().items():
        print(f"  {label}: {count} ({100 * count / len(demo):.1f}%)")

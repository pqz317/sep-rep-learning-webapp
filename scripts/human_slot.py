"""Read the human's game slot from a gameplay record stream.

Since 2026-09-20 the web app fixes human=agent 0 / model=agent 1 and writes the
assignment into every record's ``metadata['block_metadata']['human_id']`` (see
``web_app/constants.py``).  Data collected before then — everything in
``prolific_data/newflydata`` — had the slot drawn randomly per stage by nicewebrl's
``MultiAgentEnvStage`` and never logged, so it has to be recovered by replay; the
replay scripts each keep their own inference routine for that case and call it only
when this helper returns None.
"""


def recorded_human_agent(records):
    """Return the logged human slot, or None if this data predates the logging.

    Args:
        records: gameplay records as returned by
            ``nicewebrl.utils.read_all_records_sync`` (str-keyed dicts).

    Returns:
        0 or 1 if every stage record carries the same ``human_id``, else None —
        the key being absent from a stage record, or stage records disagreeing,
        both mean the slot is not trustworthy and must be inferred instead.

    Records without ``block_metadata`` at all are not stage records (gameplay
    files carry a trailing bookkeeping record) and are ignored.
    """
    human_id = None
    for record in records:
        if not isinstance(record, dict):
            continue
        block_metadata = record.get("metadata", {}).get("block_metadata")
        if not block_metadata:
            continue
        value = block_metadata.get("human_id")
        if value is None:
            return None
        if human_id is None:
            human_id = int(value)
        elif int(value) != human_id:
            return None
    return human_id

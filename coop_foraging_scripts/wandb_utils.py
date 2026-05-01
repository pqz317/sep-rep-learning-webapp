from typing import Any, Sequence

import wandb


def as_clean_str_list(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    return [str(item).strip() for item in raw if str(item).strip() != '']


def normalize_run_id(run_id: str) -> str:
    normalized = run_id.strip()
    if normalized == '':
        raise ValueError('Encountered an empty run_id.')
    return normalized


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def get_runs_for_tag(tag: str, wandb_entity: str, wandb_project: str) -> list[object]:
    api = wandb.Api()
    runs_iter = api.runs(
        f'{wandb_entity}/{wandb_project}',
        filters={'tags': {'$in': [tag]}},
    )
    return [run for run in runs_iter if tag in (run.tags or [])]


def resolve_unique_run_id_for_tag(tag: str, wandb_entity: str, wandb_project: str) -> str:
    matched_runs = get_runs_for_tag(tag=tag, wandb_entity=wandb_entity, wandb_project=wandb_project)

    if len(matched_runs) == 0:
        raise ValueError(f"No runs found for tag '{tag}' in W&B project '{wandb_entity}/{wandb_project}'.")
    if len(matched_runs) > 1:
        matched_ids = [run.id for run in matched_runs]
        raise ValueError(
            f"Tag '{tag}' matches multiple run_ids in W&B project '{wandb_entity}/{wandb_project}': {matched_ids}. "
            'Please use a unique tag or pass run_ids directly.'
        )

    resolved_run_id = normalize_run_id(matched_runs[0].id)
    print(f"Resolved tag '{tag}' -> run_id '{resolved_run_id}'")
    return resolved_run_id


def resolve_run_ids(
    run_ids: Sequence[str],
    tags: Sequence[str],
    wandb_entity: str,
    wandb_project: str,
    missing_wandb_context_error: str,
) -> list[str]:
    normalized_run_ids = [normalize_run_id(run_id) for run_id in run_ids]
    cleaned_tags = [tag.strip() for tag in tags if tag.strip() != '']

    resolved_run_ids: list[str] = list(normalized_run_ids)
    if cleaned_tags:
        if wandb_entity.strip() == '' or wandb_project.strip() == '':
            raise ValueError(missing_wandb_context_error)

        for tag in cleaned_tags:
            resolved_run_ids.append(
                resolve_unique_run_id_for_tag(
                    tag=tag,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                )
            )

    return dedupe_preserve_order(resolved_run_ids)


def resolve_run_labels_for_run_ids(
    run_ids: Sequence[str],
    preferred_tag_order: Sequence[str],
    wandb_entity: str,
    wandb_project: str,
) -> dict[str, str]:
    normalized_run_ids = [normalize_run_id(run_id) for run_id in run_ids]
    cleaned_preferred_tags = [str(tag).strip() for tag in preferred_tag_order if str(tag).strip() != '']

    run_id_to_label: dict[str, str] = {}
    api = wandb.Api()
    for run_id in normalized_run_ids:
        label = run_id
        try:
            run = api.run(f'{wandb_entity}/{wandb_project}/{run_id}')
            tags = list(run.tags or [])
            for tag in cleaned_preferred_tags:
                if tag in tags:
                    label = tag
                    break
            else:
                if len(tags) > 0:
                    label = tags[0]
        except Exception:
            label = run_id

        run_id_to_label[run_id] = label

    return run_id_to_label


def get_run_config(run_id: str, wandb_entity: str, wandb_project: str) -> dict[str, Any]:
    if wandb_entity.strip() == '' or wandb_project.strip() == '':
        raise ValueError('Cannot fetch run config without logging.wandb_entity and logging.wandb_project.')

    normalized_run_id = normalize_run_id(run_id)
    api = wandb.Api()
    run = api.run(f'{wandb_entity}/{wandb_project}/{normalized_run_id}')
    config = run.config or {}
    if not isinstance(config, dict):
        raise ValueError(
            f'Expected W&B run config to be a dict for run_id={normalized_run_id}, got {type(config).__name__}.'
        )
    return dict(config)
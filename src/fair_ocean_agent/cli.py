"""Command-line interface (section 18). init-db, ingest-seeds,
enqueue-seed-backfill, worker, status, export-raw-facts, report-run,
benchmark-models, discover, FAIRe export, weekly update, and validate
(Milestone 5: data-asset inventory + logical/evidence/cross-source
validation) are fully implemented. Commands that still depend on unbuilt
layers (the full per-study `run-study` pipeline and BeBOP export) are
present as explicit stubs -- they raise typer.Exit(1) with a clear "not yet
implemented" message rather than silently no-oping.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from fair_ocean_agent.config import load_benchmark_candidates, load_config
from fair_ocean_agent.database.enums import TaskType
from fair_ocean_agent.database.models import ValidationResult, WorkflowRun
from fair_ocean_agent.database.session import check_schema_drift as _check_schema_drift
from fair_ocean_agent.database.session import init_db as _init_db
from fair_ocean_agent.database.session import reset_database as _reset_database
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.discovery.seed_loader import enqueue_seed_backfill, ingest_seed_file
from fair_ocean_agent.exports.faire import export_faire
from fair_ocean_agent.exports.raw_facts import export_raw_facts
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.benchmark import export_report, load_gold_cases, run_benchmark
from fair_ocean_agent.llm.factory import build_benchmark_backend
from fair_ocean_agent.logging_setup import setup_logging
from fair_ocean_agent.scheduling.weekly import run_weekly_update
from fair_ocean_agent.sources.base import SearchQuery
from fair_ocean_agent.standards.registry import write_registry
from fair_ocean_agent.workflow.task_queue import count_tasks_by_status, release_stale_claims
from fair_ocean_agent.workflow.worker import DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES, run_worker
import fair_ocean_agent.workflow.handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS
import fair_ocean_agent.workflow.mapping_handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS
import fair_ocean_agent.workflow.refresh_handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS
import fair_ocean_agent.workflow.settle_handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS
import fair_ocean_agent.workflow.supplement_handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS
import fair_ocean_agent.workflow.validation_handlers  # noqa: F401 -- side effect: registers TASK_HANDLERS

app = typer.Typer(add_completion=False, help="FAIR ocean eDNA metadata curation pipeline")
console = Console()


@app.callback()
def main() -> None:
    setup_logging()


@app.command("init-db")
def init_db_command() -> None:
    """Create all tables directly from the ORM models (dev/local
    convenience). In production, use `alembic upgrade head` instead so
    schema changes are tracked as migrations."""
    _init_db()
    console.print("[green]Database initialized.[/green]")


@app.command("check-schema-drift")
def check_schema_drift_command() -> None:
    """Read-only: report whether this database was ever fully brought up to
    date via `alembic upgrade head`, and if not, exactly which columns
    (added by a migration to an already-existing table, which init-db's
    create_all() can never backfill) are missing from it right now. Changes
    nothing -- safe to run against a real, populated database at any time."""
    report = _check_schema_drift()
    if report["alembic_version_table_present"] and not report["missing_columns_by_table"]:
        console.print("[green]No drift detected: alembic_version is present and every table has all current columns.[/green]")
        console.print("`alembic upgrade head` is safe to run directly.")
        return

    if not report["alembic_version_table_present"]:
        console.print("[yellow]No alembic_version table -- this database has never been Alembic-managed "
                       "(likely bootstrapped via init-db only).[/yellow]")

    missing = report["missing_columns_by_table"]
    if missing:
        console.print("[yellow]Columns present in the ORM models but missing from this database:[/yellow]")
        for table_name, columns in missing.items():
            console.print(f"  {table_name}: {', '.join(columns)}")
    else:
        console.print("[green]No missing columns detected on any existing table.[/green]")

    console.print(
        "\n[bold]Do not run `alembic upgrade head` directly yet[/bold] -- without an "
        "alembic_version row it will try to CREATE TABLE for tables that already "
        "exist and fail. Report this output before applying a fix."
    )


@app.command("reset-database")
def reset_database_command(
    yes: bool = typer.Option(
        False, "--yes", help="Required: confirms permanent erasure of every study/entity/fact/task in this database."
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="Skip the automatic timestamped backup copy (sqlite only)."
    ),
) -> None:
    """DESTRUCTIVE. Drops every table in this database and rebuilds an
    empty schema via `alembic upgrade head` -- every study, entity, fact,
    source, and task is gone. For active development/debugging only: lets
    you re-run every seed paper from scratch against current code instead
    of debugging one study's state, accumulated across many discovery-logic
    changes, at a time.

    Does NOT touch data/cache/ (the on-disk HTTP response cache) or any
    local/auto-fetched PDFs -- re-ingesting afterward stays fast because
    those upstream responses are still cached; only the derived database
    state resets. For sqlite, makes a timestamped backup file first unless
    --no-backup is passed.

    After this completes: `ingest-seeds <file>`, `enqueue-seed-backfill`,
    `worker --until-empty`, same as any first run."""
    if not yes:
        console.print(
            "[red]Refusing to run without --yes.[/red] This permanently erases every "
            "study, entity, fact, source, and task in the database."
        )
        raise typer.Exit(1)
    backup_path = _reset_database(backup=not no_backup)
    if backup_path is not None:
        console.print(f"[green]Backed up existing database to {backup_path}[/green]")
    console.print("[green]Database reset: schema rebuilt via alembic upgrade head, all data erased.[/green]")
    console.print("Next: ingest-seeds <file>, enqueue-seed-backfill, worker --until-empty")


@app.command("ingest-seeds")
def ingest_seeds_command(
    path: Path = typer.Argument(..., help="Path to a seed CSV or JSONL file"),
) -> None:
    with session_scope() as session:
        results = ingest_seed_file(session, path)

    created = sum(1 for r in results if r.created)
    skipped = sum(1 for r in results if r.skipped)
    merged = len(results) - created - skipped
    errors = sum(len(r.identifier_errors) for r in results)

    console.print(f"Ingested {len(results)} seed rows: {created} new studies, {merged} merged, {skipped} skipped (excluded DOI).")
    if errors:
        console.print(f"[yellow]{errors} identifier(s) failed normalization (see below).[/yellow]")
        for r in results:
            for err in r.identifier_errors:
                console.print(f"  [yellow]{r.seed_id or '?'}: {err}[/yellow]")


@app.command("enqueue-seed-backfill")
def enqueue_seed_backfill_command() -> None:
    with session_scope() as session:
        count = enqueue_seed_backfill(session)
    console.print(f"Queued DISCOVER_IDENTIFIERS for {count} candidate stud(y/ies).")


@app.command("enqueue-text-extraction-backfill")
def enqueue_text_extraction_backfill_command() -> None:
    """Queue EXTRACT_TEXT_FACTS for every study with a known PMCID.
    Separate/opt-in from enqueue-seed-backfill -- with llm.enabled: false
    (the default), these tasks will fail against a disabled LLM backend
    until you configure a real endpoint in config/default.yaml's llm:
    section."""
    with session_scope() as session:
        count = fair_ocean_agent.workflow.handlers.enqueue_text_extraction_backfill(session)
    console.print(f"Queued EXTRACT_TEXT_FACTS for {count} stud(y/ies) with a known PMCID.")


@app.command("enqueue-supplement-discovery-backfill")
def enqueue_supplement_discovery_backfill_command() -> None:
    """Queue DISCOVER_SUPPLEMENTS for every study with a known PMCID --
    parses <supplementary-material> references out of the (cached) main
    article XML; no new network call. Run before
    enqueue-supplement-retrieval-backfill."""
    with session_scope() as session:
        count = fair_ocean_agent.workflow.supplement_handlers.enqueue_supplement_discovery_backfill(session)
    console.print(f"Queued DISCOVER_SUPPLEMENTS for {count} stud(y/ies) with a known PMCID.")


@app.command("enqueue-supplement-retrieval-backfill")
def enqueue_supplement_retrieval_backfill_command() -> None:
    """Queue RETRIEVE_SUPPLEMENTS for every study with at least one
    discovered supplement (run enqueue-supplement-discovery-backfill
    first). Fetches Europe PMC's supplementary-files bundle (size-capped)
    and deterministically parses CSV/TSV/XLSX/XLS/JSON/XML members; PDF/TXT/MD
    go through the existing LLM extraction path; anything else is
    inventoried only."""
    with session_scope() as session:
        count = fair_ocean_agent.workflow.supplement_handlers.enqueue_supplement_retrieval_backfill(session)
    console.print(f"Queued RETRIEVE_SUPPLEMENTS for {count} stud(y/ies) with a discovered supplement.")


@app.command("enqueue-supplement-text-extraction-backfill")
def enqueue_supplement_text_extraction_backfill_command() -> None:
    """Queue the opt-in, post-paper LLM pass over prepared supplement text.

    Retrieval prepares TXT/Markdown/PDF text without invoking a model.
    This command is the explicit switch for the later missing-field-only
    extraction stage.
    """
    config = load_config()
    if not config.supplements.llm_text_extraction_enabled:
        console.print(
            "[yellow]Supplement text LLM extraction is disabled.[/yellow] "
            "Set FAIR_OCEAN_SUPPLEMENT_LLM_ENABLED=true (or "
            "supplements.llm_text_extraction_enabled: true in config/local.yaml) "
            "when you are ready to test it."
        )
        return
    with session_scope() as session:
        count = fair_ocean_agent.workflow.supplement_handlers.enqueue_supplement_text_extraction_backfill(
            session
        )
    console.print(
        f"Queued EXTRACT_SUPPLEMENT_TEXT_FACTS for {count} paper-ready stud(y/ies)."
    )


@app.command("enqueue-validation-backfill")
def enqueue_validation_backfill_command() -> None:
    """Queue INVENTORY_DATA_ASSETS, VALIDATE_LOGIC, VALIDATE_EVIDENCE, and
    VALIDATE_CROSS_SOURCE for every study that has at least one raw_fact."""
    with session_scope() as session:
        counts = fair_ocean_agent.workflow.validation_handlers.enqueue_data_asset_and_validation_backfill(session)
    console.print(f"Queued: {counts}")


@app.command("enqueue-mapping-backfill")
def enqueue_mapping_backfill_command() -> None:
    """Queue MAP_FAIRE for every study that has at least one raw_fact."""
    with session_scope() as session:
        count = fair_ocean_agent.workflow.mapping_handlers.enqueue_mapping_backfill(session)
    console.print(f"Queued MAP_FAIRE for {count} stud(y/ies).")


@app.command("enqueue-faire-completeness-backfill")
def enqueue_faire_completeness_backfill_command() -> None:
    """Queue VALIDATE_FAIRE_COMPLETENESS for every study that already has at
    least one FAIRe StandardizedValue (run enqueue-mapping-backfill first)."""
    with session_scope() as session:
        count = fair_ocean_agent.workflow.validation_handlers.enqueue_faire_completeness_backfill(session)
    console.print(f"Queued VALIDATE_FAIRE_COMPLETENESS for {count} stud(y/ies).")


@app.command("enqueue-full-rediscovery-backfill")
def enqueue_full_rediscovery_backfill_command() -> None:
    """Re-run DISCOVER_IDENTIFIERS for every candidate study, on demand --
    not just ones with a failed/manual-review task. The concrete case this
    catches: a newly-enabled adapter (or newly-supported identifier type,
    e.g. Zenodo/Dryad/Figshare/OSF, SRA run-level accessions, the
    open-access PDF auto-fetch) that already-discovered studies were never
    checked against -- DISCOVER_IDENTIFIERS's own idempotency otherwise
    means a study already discovered once is never revisited, even after
    the discovery logic itself gains new capability. `weekly-update` also
    runs this automatically every quarterly_full_rediscovery_interval_days
    (config.py's SchedulingConfig); this command is for triggering it
    immediately instead of waiting for that cadence. Excludes any study
    already marked data_availability_status=not_accessible (give-up
    tracking, see workflow/handlers.py's _has_accessible_sequence_data_signal)
    -- if the last full check found nothing accessible anywhere, a newly-
    enabled adapter is exactly the kind of thing that verdict should be
    re-checked against, so consider clearing that flag for studies you
    specifically want re-examined rather than skipped."""
    from fair_ocean_agent.clock import utcnow
    from fair_ocean_agent.database.enums import WorkflowRunStatus
    from fair_ocean_agent.scheduling.rediscovery import RUN_TYPE, enqueue_full_rediscovery

    with session_scope() as session:
        run = WorkflowRun(run_type=RUN_TYPE, status=WorkflowRunStatus.RUNNING.value)
        session.add(run)
        session.flush()
        count = enqueue_full_rediscovery(session, run.run_id)
        run.candidates_found = count
        run.status = WorkflowRunStatus.COMPLETED.value
        run.ended_at = utcnow()
    console.print(f"Queued DISCOVER_IDENTIFIERS for {count} candidate stud(y/ies).")


@app.command("enqueue-citation-rediscovery-backfill")
def enqueue_citation_rediscovery_backfill_command() -> None:
    """Node-adding discovery, run on demand: re-checks every already-known
    BioProject accession for NEW citing papers (workflow/handlers.py's
    DISCOVER_CITING_STUDIES is normally only triggered once per accession,
    the first time it's resolved -- this is what catches a paper published
    or indexed by PubMed AFTER that first resolution). `weekly-update` also
    runs this automatically every citation_rediscovery_interval_days
    (config.py's SchedulingConfig); this command is for triggering it
    immediately instead of waiting for that cadence."""
    from fair_ocean_agent.clock import utcnow
    from fair_ocean_agent.database.enums import WorkflowRunStatus
    from fair_ocean_agent.scheduling.rediscovery import enqueue_citation_rediscovery_backfill

    with session_scope() as session:
        run = WorkflowRun(run_type="citation_rediscovery", status=WorkflowRunStatus.RUNNING.value)
        session.add(run)
        session.flush()
        count = enqueue_citation_rediscovery_backfill(session, run.run_id)
        run.candidates_found = count
        run.status = WorkflowRunStatus.COMPLETED.value
        run.ended_at = utcnow()
    console.print(f"Queued DISCOVER_CITING_STUDIES for {count} BioProject accession(s).")


@app.command("enqueue-full-text-extraction-backfill")
def enqueue_full_text_extraction_backfill_command() -> None:
    """Re-run EXTRACT_TEXT_FACTS for every study with a known PMCID, on
    demand -- not just ones that never had one. The concrete case this
    catches: a newly-added or fixed text-extraction capability (e.g.
    extract_primer_reference_citations' primer-reference chasing) that
    already-processed studies were never checked against --
    EXTRACT_TEXT_FACTS's own idempotency otherwise means a study already
    extracted once is never revisited, even after the extraction logic
    itself gains new capability. Same shape as enqueue-full-rediscovery-
    backfill, one layer down the pipeline -- see
    scheduling/rediscovery.py::enqueue_full_text_extraction_backfill's own
    docstring for the confirmed-live example that motivated this."""
    from fair_ocean_agent.clock import utcnow
    from fair_ocean_agent.database.enums import WorkflowRunStatus
    from fair_ocean_agent.scheduling.rediscovery import TEXT_EXTRACTION_RUN_TYPE, enqueue_full_text_extraction_backfill

    with session_scope() as session:
        run = WorkflowRun(run_type=TEXT_EXTRACTION_RUN_TYPE, status=WorkflowRunStatus.RUNNING.value)
        session.add(run)
        session.flush()
        count = enqueue_full_text_extraction_backfill(session, run.run_id)
        run.candidates_found = count
        run.status = WorkflowRunStatus.COMPLETED.value
        run.ended_at = utcnow()
    console.print(f"Queued EXTRACT_TEXT_FACTS for {count} stud(y/ies) with a known PMCID.")


# The only three task types that ever call an LLM backend at all
# (confirmed by reading every TASK_HANDLERS registration, not guessed):
# EXTRACT_TEXT_FACTS and EXTRACT_SUPPLEMENT_TEXT_FACTS call the main
# extraction backend; VALIDATE_EVIDENCE calls the (usually smaller)
# verifier backend. Every other task type -- DISCOVER_IDENTIFIERS
# included -- is pure deterministic regex/API work, no model involved at
# any point. Per an explicit user request: a machine with no GPU worth
# running local extraction on needs an easy, durable way to guarantee a
# `worker` invocation never claims one of these three, without having to
# enumerate every other (and every future) task type by hand.
_LLM_TASK_TYPES = (TaskType.EXTRACT_TEXT_FACTS, TaskType.EXTRACT_SUPPLEMENT_TEXT_FACTS, TaskType.VALIDATE_EVIDENCE)


@app.command("worker")
def worker_command(
    max_tasks: int | None = typer.Option(None, help="Stop after processing this many tasks"),
    until_empty: bool = typer.Option(False, help="Process tasks until the queue is empty"),
    worker_id: str = typer.Option("local-worker", help="Identifier recorded on claimed tasks"),
    task_type: list[str] = typer.Option(
        None,
        "--task-type",
        help="Only claim this task type (repeatable, e.g. --task-type DISCOVER_IDENTIFIERS "
        "--task-type DISCOVER_SUPPLEMENTS). Omit to claim any type.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Never claim EXTRACT_TEXT_FACTS, EXTRACT_SUPPLEMENT_TEXT_FACTS, or VALIDATE_EVIDENCE "
        "-- the only three task types that call a model at all. For a machine (e.g. a laptop "
        "with no GPU worth running local extraction on) that should only ever do the CPU-only "
        "discovery/mapping/other-validation work and leave the LLM stage to the cluster.",
    ),
    max_consecutive_429s: int = typer.Option(
        DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
        help="Stop the whole run (not just one task) once this many tasks in a row fail with "
        "429 Too Many Requests -- protects against hammering a source that's actively "
        "rate-limiting/blocking this machine across a deep queue. Resets on any success or "
        "non-429 failure. Set to 0 to disable (never stop early).",
    ),
) -> None:
    if task_type and no_llm:
        console.print("[red]--task-type and --no-llm are mutually exclusive -- pick one.[/red]")
        raise typer.Exit(1)
    task_types: list[TaskType] | None = None
    if task_type:
        try:
            task_types = [TaskType(t) for t in task_type]
        except ValueError as exc:
            valid = ", ".join(t.value for t in TaskType)
            console.print(f"[red]{exc}[/red]\nValid task types: {valid}")
            raise typer.Exit(1)
    elif no_llm:
        task_types = [t for t in TaskType if t not in _LLM_TASK_TYPES]

    try:
        with session_scope() as session:
            summary = run_worker(
                session,
                worker_id=worker_id,
                max_tasks=max_tasks,
                until_empty=until_empty,
                task_types=task_types,
                max_consecutive_rate_limit_failures=max_consecutive_429s if max_consecutive_429s > 0 else None,
            )
    finally:
        fair_ocean_agent.workflow.handlers.reset_adapter_cache()
        fair_ocean_agent.workflow.handlers.reset_llm_backend_cache()
    console.print(
        f"Processed {summary['processed']} task(s): "
        f"{summary['completed']} completed, {summary['failed']} failed."
    )
    if summary["stopped_reason"]:
        console.print(f"[red]Stopped early:[/red] {summary['stopped_reason']}")
        # Distinct, non-zero exit so a `worker && next_step` chain (or any
        # automation) stops here instead of walking straight into the next
        # network-calling stage right after confirming a source is actively
        # rate-limiting/blocking this machine -- exit code 2 matches
        # run_mgnify_seed_discovery.py/run_ena_seed_discovery.py's own
        # convention for the same "stopped due to sustained 429" case.
        raise typer.Exit(2)


@app.command("run-study")
def run_study_command(study_id: str = typer.Argument(...)) -> None:
    console.print(
        "[yellow]Not yet implemented:[/yellow] running the full per-study pipeline "
        "requires source adapters, extraction, mapping, and validation "
        "(Milestones 2-6). Use `enqueue-seed-backfill` + `worker` to exercise "
        f"the task queue for {study_id} today."
    )
    raise typer.Exit(1)


@app.command("discover")
def discover_command(
    source: str = typer.Option(..., help="Adapter name, e.g. datacite, gbif, obis, bcodmo, pangaea"),
    query: str = typer.Option(..., help="Search query to send to the selected adapter"),
    limit: int = typer.Option(20, help="Maximum records to print"),
) -> None:
    """Run a bounded live search against one enabled source adapter and
    print returned records. This is an API smoke/discovery command only; use
    `ingest-seeds` + `enqueue-seed-backfill` + `worker` to persist sources
    and raw_facts for studies."""
    adapter_name = source.replace("-", "_").lower()
    adapters = fair_ocean_agent.workflow.handlers._build_enabled_adapters()
    adapter = adapters.get(adapter_name)
    if adapter is None:
        available = ", ".join(sorted(adapters))
        console.print(f"[red]Source adapter '{source}' is not enabled. Available: {available}[/red]")
        raise typer.Exit(1)

    try:
        page = adapter.search(SearchQuery(query=query, limit=limit))
    finally:
        fair_ocean_agent.workflow.handlers.reset_adapter_cache()

    table = Table(title=f"{adapter_name} search: {query}")
    for column in ("source", "identifier", "url"):
        table.add_column(column)
    for record in page.records:
        table.add_row(record.source_name, record.external_identifier, record.url or "")
    console.print(table)
    total = page.total_count if page.total_count is not None else len(page.records)
    console.print(f"Returned {len(page.records)} record(s); total reported: {total}.")


@app.command("validate")
def validate_command(study_id: str = typer.Option(...)) -> None:
    """Shows already-recorded ValidationResult rows for a study (logical,
    evidence-consistency, cross-source). Run `enqueue-validation-backfill` +
    `worker` first to actually generate them -- this command only displays
    what's been persisted, it doesn't run validators itself."""
    with session_scope() as session:
        results = (
            session.query(ValidationResult)
            .filter_by(study_id=study_id)
            .order_by(ValidationResult.severity.desc(), ValidationResult.created_at)
            .all()
        )

    if not results:
        console.print(
            f"No validation results recorded for {study_id}. Run "
            "`fair-ocean enqueue-validation-backfill` then `fair-ocean worker` first."
        )
        return

    table = Table(title=f"Validation results for {study_id}")
    for column in ("validator", "status", "severity", "message"):
        table.add_column(column)
    for r in results:
        style = "red" if r.severity == "error" else ("yellow" if r.severity == "warning" else None)
        table.add_row(r.validator_name, r.status, r.severity, r.message[:100], style=style)
    console.print(table)

    review_needed = [r for r in results if r.status in ("conflicting", "unsupported")]
    if review_needed:
        console.print(f"[yellow]{len(review_needed)} result(s) need manual review (conflicting/unsupported).[/yellow]")


@app.command("export-raw-facts")
def export_raw_facts_command(
    output: Path = typer.Option(..., help="Output CSV path"),
) -> None:
    with session_scope() as session:
        n = export_raw_facts(session, output)
    console.print(f"Wrote {n} raw fact(s) to {output}")


@app.command("export-faire")
def export_faire_command(
    output: Path = typer.Option(..., help="Output directory; one CSV per FAIRe class is written here"),
) -> None:
    """Exports FAIRe StandardizedValues (run `enqueue-mapping-backfill` +
    `worker` first to generate them) as one CSV per FAIRe class, matching
    the FULLtemplate.xlsx sheet layout. See exports/faire.py's docstring
    for which classes have real data vs. header-only placeholders."""
    with session_scope() as session:
        counts = export_faire(session, output)
    for class_name, count in counts.items():
        console.print(f"Wrote {count} row(s) to {output / (class_name + '.csv')}")


@app.command("build-standards-registry")
def build_standards_registry_command(
    output_dir: Path = typer.Option(Path("standards/compiled"), help="Directory to write the compiled registry files into"),
) -> None:
    """Compiles schemas/faire/, schemas/miop/, and schemas/bebop/ into the
    normalized standards/ registry (faire_registry.json,
    bebop_miop_registry.json, term_crosswalk.csv, template_field_usage.csv,
    standards_validation_report.json). Does not touch this pipeline's own
    database -- purely a schema-compilation step, safe to re-run any time
    a vendored schema file changes."""
    counts = write_registry(output_dir)
    console.print(f"Wrote standards registry to {output_dir}: {counts}")

    import json

    report = json.loads((output_dir / "standards_validation_report.json").read_text())
    failed = [name for name, result in report["checks"].items() if not result["passed"]]
    if failed:
        console.print(f"[red]{len(failed)} validation check(s) failed: {failed}[/red]")
        raise typer.Exit(1)
    console.print("[green]All standards registry validation checks passed.[/green]")


@app.command("export-bebop")
def export_bebop_command(output: Path = typer.Option(...)) -> None:
    console.print(
        "[yellow]Out of scope:[/yellow] BeBOP/MIOP describes protocol-"
        "document metadata (author, license, version), not facts extracted "
        "from published papers -- this pipeline's current input type. The "
        "miop/BeBOP schemas are vendored and compiled (see `fair-ocean "
        "build-standards-registry`); mapping paper-derived raw_facts onto "
        "them would fabricate metadata papers don't carry. See "
        "mapping/bebop.py's module docstring -- this would be revisited if "
        "this pipeline ever ingests an actual protocol submission."
    )
    raise typer.Exit(1)


@app.command("weekly-update")
def weekly_update_command(dry_run: bool = typer.Option(False, help="Report what would happen without changing anything")) -> None:
    """Enqueue-only, like every other enqueue-* command: refreshes known
    studies' sources, retries long-stale failed/manual-review tasks, and
    (quarterly) re-runs DISCOVER_IDENTIFIERS for every study. Run
    `fair-ocean worker --until-empty` separately to actually process what
    this queues."""
    with session_scope() as session:
        run = run_weekly_update(session, dry_run=dry_run)
        run_id, summary = run.run_id, run.sources_queried
        console.print(f"Run {run_id} ({'dry-run' if dry_run else 'live'}): {summary}")


@app.command("benchmark-models")
def benchmark_models_command(
    gold_dir: Path = typer.Option(Path("data/benchmark/gold"), help="Directory of gold-case JSON files"),
    output: Path = typer.Option(Path("data/exports/benchmark"), help="Output directory for summary.csv/detail.json"),
) -> None:
    """Run every candidate in config/benchmark_models.yaml against the gold
    cases in gold_dir and write a comparison report. Every candidate must
    point at a real, currently-running inference endpoint -- this command
    does not choose or fall back to any model on its own; a candidate left
    at its placeholder values raises immediately."""
    cases = load_gold_cases(gold_dir)
    if not cases:
        console.print(f"[red]No gold cases found under {gold_dir}[/red]")
        raise typer.Exit(1)

    candidates = load_benchmark_candidates()
    if not candidates:
        console.print("[red]No candidates configured in config/benchmark_models.yaml[/red]")
        raise typer.Exit(1)

    backends = []
    try:
        for candidate in candidates:
            try:
                backends.append(build_benchmark_backend(candidate))
            except LLMBackendError as exc:
                console.print(f"[yellow]Skipping {candidate.label}:[/yellow] {exc}")

        if not backends:
            console.print("[red]No usable candidates -- fill in config/benchmark_models.yaml.[/red]")
            raise typer.Exit(1)

        console.print(f"Running {len(cases)} gold case(s) against {len(backends)} candidate(s)...")
        report = run_benchmark(backends, cases)
    finally:
        for backend in backends:
            backend.close()

    export_report(report, output)

    table = Table(title="Benchmark summary")
    for column in ("candidate", "n", "json_valid", "evidence_verified", "precision", "recall", "f1", "mean_latency_s"):
        table.add_column(column)
    for label, (metrics, _results) in report.items():
        table.add_row(
            label,
            str(metrics.n_cases),
            f"{metrics.json_validity_rate:.2f}",
            f"{metrics.evidence_verification_rate:.2f}",
            f"{metrics.precision:.2f}",
            f"{metrics.recall:.2f}",
            f"{metrics.f1:.2f}",
            f"{metrics.mean_latency_seconds:.2f}",
        )
    console.print(table)
    console.print(f"Full report written to {output}/summary.csv and {output}/detail.json")


@app.command("status")
def status_command(as_json: bool = typer.Option(False, "--json")) -> None:
    with session_scope() as session:
        task_counts = count_tasks_by_status(session)

    if as_json:
        console.print(json.dumps(task_counts))
        return

    table = Table(title="Task queue status")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in task_counts.items():
        table.add_row(status, str(count))
    console.print(table)


@app.command("release-stale-claims")
def release_stale_claims_command(
    stale_after_minutes: int = typer.Option(30),
) -> None:
    with session_scope() as session:
        n = release_stale_claims(session, stale_after_minutes)
    console.print(f"Released {n} stale claim(s).")


@app.command("report-run")
def report_run_command(run_id: str = typer.Argument(...)) -> None:
    with session_scope() as session:
        run = session.get(WorkflowRun, run_id)
    if run is None:
        console.print(f"[red]No workflow run found with id {run_id}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Run {run_id}")
    table.add_column("Field")
    table.add_column("Value")
    for column in run.__table__.columns:
        table.add_row(column.name, str(getattr(run, column.name)))
    console.print(table)


if __name__ == "__main__":
    app()

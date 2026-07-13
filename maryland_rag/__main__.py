"""
Maryland Elections RAG Pipeline entry point.

Usage:
    python -m maryland_rag pass1              # Run Pass 1 crawl (BoE + MoCo)
    python -m maryland_rag pass1 --no-resume  # Start fresh instead of resuming
    python -m maryland_rag pass2              # Run Pass 2 chunking (all pages)
    python -m maryland_rag pass2 --changed    # Only re-chunk changed pages
    python -m maryland_rag pass3              # Embed chunks and insert into pgvector
    python -m maryland_rag pass3 --resume     # Skip already-embedded chunk IDs
    python -m maryland_rag audit              # Print manifest audit report
    python -m maryland_rag all                # Full pipeline end-to-end:
                                              #   pass1 → pass2 → box_ingest → pass3 (web) → pass3 (box)
                                              #   pass2 ALWAYS re-chunks every crawled page (never
                                              #   incremental): the operator drops the pgvector DB and
                                              #   re-ingests from scratch each run, and pass2 truncates
                                              #   chunks.jsonl, so it must contain the full corpus.
                                              #   Incremental chunking is only for `pass2 --changed`.
"""
import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _configure_logging():
    from .pass1.config import LOG_DIR, LOG_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


def _report_chunk_coverage(chunks: list[dict], expected_urls: set[str]):
    """
    Compare the pages that actually produced chunks against the pages that
    should have (crawl_status='crawled', not strategy='skip'). Zero-chunk
    pages are printed loudly; if more than 20% of expected pages produced
    nothing, exit non-zero — the operator rebuilds pgvector from
    chunks.jsonl, so shipping a badly incomplete file would silently drop
    that content from the vector store.
    """
    covered = set()
    for chunk in chunks:
        if chunk.get('source_url'):
            covered.add(chunk['source_url'])
        # Deduplicated content carries every URL it appeared at
        covered.update(chunk.get('source_urls', []))

    missing = sorted(expected_urls - covered)
    n_expected = len(expected_urls)
    print(f"Chunk coverage: {n_expected - len(missing)}/{n_expected} expected pages produced chunks")

    if missing:
        logger.warning("%d expected page(s) produced ZERO chunks:", len(missing))
        print(f"WARNING: {len(missing)} expected page(s) produced ZERO chunks:")
        for url in missing:
            logger.warning("  zero chunks: %s", url)
            print(f"  ZERO CHUNKS: {url}")

    if n_expected and len(missing) > 0.2 * n_expected:
        msg = (
            f"FATAL: {len(missing)}/{n_expected} expected pages "
            f"({len(missing) / n_expected:.0%}) produced zero chunks — aborting "
            "before embedding so pgvector is not rebuilt from an incomplete "
            "chunks.jsonl."
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        sys.exit(1)


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description='Maryland Elections RAG Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Pipeline pass to run')

    # Pass 1
    p1 = subparsers.add_parser('pass1', help='Crawl and map the elections site')
    p1.add_argument('--no-resume', action='store_true', help='Start fresh instead of resuming')

    # Pass 2
    p2 = subparsers.add_parser('pass2', help='Extract and chunk all crawled pages')
    p2.add_argument('--changed', action='store_true', help='Only process pages that changed')
    p2.add_argument('--output', default='data/chunks.jsonl', help='Output JSONL path')

    # Pass 3
    p3 = subparsers.add_parser('pass3', help='Embed chunks and insert into pgvector')
    p3.add_argument('--chunks', default='data/chunks.jsonl', help='Path to chunks JSONL')
    p3.add_argument('--resume', action='store_true', help='Skip already-upserted chunk IDs')

    # All Passes
    all = subparsers.add_parser('all', help='Run all passes sequentially, resuming from previous runs')

    all.add_argument('--output', default='data/chunks.jsonl', help='Output JSONL path for web chunks')

    # Audit
    subparsers.add_parser('audit', help='Print manifest audit report')

    args = parser.parse_args()

    if args.command == 'pass1':
        from .pass1.crawler import run_crawl
        run_crawl(resume=not args.no_resume)

    elif args.command == 'pass2':
        from .pass2.chunker import run_pass2
        chunks = run_pass2(only_changed=args.changed, output_path=args.output)
        print(f"Produced {len(chunks)} chunks → {args.output}")
        if args.changed:
            print(
                "WARNING: --changed wrote ONLY the changed pages' chunks to "
                f"{args.output} (the file is truncated, not merged). Do not "
                "embed it into a freshly-dropped database — use the full "
                "`all` pipeline (or plain `pass2`) for a from-scratch rebuild."
            )

    elif args.command == 'pass3':
        from .pass3.embed import run_embed
        n = run_embed(chunks_path=args.chunks, resume=args.resume)
        print(f"Inserted {n} vectors into pgvector")

    elif args.command == 'all':
        from .pass1.crawler import run_crawl
        from .pass2.chunker import run_pass2
        from .pass3.embed import run_embed
        from .pass1.db import DB

        run_crawl(resume=False)

        # INVARIANT: 'all' always chunks the FULL corpus (only_changed=False).
        # The operator drops the pgvector DB and re-ingests from scratch each
        # run, and pass2 truncates chunks.jsonl — an incremental pass would
        # silently drop every unchanged page from the rebuilt vector store
        # (manifest.db persists, so change detection would find almost
        # nothing "changed"). Incremental chunking remains available via the
        # standalone `pass2 --changed` subcommand.
        chunks = run_pass2(only_changed=False, output_path=args.output)
        print(f"Produced {len(chunks)} chunks → {args.output}")

        db = DB()
        # Coverage gate: exits non-zero if >20% of expected pages got no chunks.
        _report_chunk_coverage(chunks, db.get_expected_chunk_urls())
        # Snapshot AFTER a successful full chunk pass: previous_content_hash
        # now means "this content is reflected in chunks.jsonl", which is the
        # baseline `pass2 --changed` diffs against. Snapshotting before the
        # crawl (the old design) permanently swallowed change events whenever
        # a run was interrupted between crawl and chunking.
        db.snapshot_hashes_for_recrawl()
        db.close()

        from box_ingest.ingest import run_ingest, DEFAULT_OUTPUT as BOX_OUTPUT
        box_chunks = run_ingest()
        print(f"Box ingest: {len(box_chunks)} chunks")

        n = run_embed(chunks_path=args.output, resume=False)
        print(f"Inserted {n} web vectors into pgvector")

        if box_chunks:
            n_box = run_embed(chunks_path=str(BOX_OUTPUT), resume=False)
            print(f"Inserted {n_box} Box vectors into pgvector")

    elif args.command == 'audit':
        from .scripts.audit import run_audit
        run_audit()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

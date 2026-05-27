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
                                              #   First run: processes everything
                                              #   Subsequent runs: only changed/new pages re-chunked;
                                              #   pass3 always upserts (resume=False) so changed
                                              #   content is never skipped in the vector DB
"""
import argparse
import logging
import os
import sys


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

    elif args.command == 'pass3':
        from .pass3.embed import run_embed
        n = run_embed(chunks_path=args.chunks, resume=args.resume)
        print(f"Inserted {n} vectors into pgvector")

    elif args.command == 'all':
        from .pass1.crawler import run_crawl
        from .pass2.chunker import run_pass2
        from .pass3.embed import run_embed
        from .pass1.db import DB

        db = DB()
        # INVARIANT: first_run must be captured BEFORE snapshot_hashes_for_recrawl().
        # is_first_run() returns True only when no content hashes exist; once the
        # snapshot runs, previous_content_hash is populated and the question
        # becomes meaningless. The captured value is then passed to pass2 as
        # only_changed=not first_run so a first-ever run chunks everything and
        # subsequent runs only re-chunk pages whose content actually changed.
        first_run = db.is_first_run()
        db.snapshot_hashes_for_recrawl()
        db.close()

        run_crawl(resume=False)

        chunks = run_pass2(only_changed=not first_run, output_path=args.output)
        print(f"Produced {len(chunks)} chunks → {args.output}")

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

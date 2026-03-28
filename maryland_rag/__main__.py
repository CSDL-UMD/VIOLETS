"""
Maryland Elections RAG Pipeline entry point.

Usage:
    python -m maryland_rag pass1              # Run Pass 1 crawl
    python -m maryland_rag pass1 --resume     # Resume interrupted Pass 1
    python -m maryland_rag pass2              # Run Pass 2 chunking
    python -m maryland_rag pass2 --changed    # Only re-chunk changed pages
    python -m maryland_rag audit              # Print manifest audit queries
    python -m maryland_rag all                # Run all passes, updates with any new links
    
"""
import argparse
import sys


def main():
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
    p3 = subparsers.add_parser('pass3', help='Embed chunks and upsert to Pinecone')
    p3.add_argument('--chunks', default='data/chunks.jsonl', help='Path to chunks JSONL')
    p3.add_argument('--resume', action='store_true', help='Skip already-upserted chunk IDs')

    # All Passes
    all = subparsers.add_parser('all', help='Run all passes sequentially, resuming from previous runs')

    all.add_argument('--output', default='data/chunks.jsonl', help='Output JSONL path')
    all.add_argument('--chunks', default='data/chunks.jsonl', help='Path to chunks JSONL')

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
        print(f"Upserted {n} vectors to Pinecone")

    elif args.command == 'all':
        from .pass1.crawler import run_crawl
        from .pass2.chunker import run_pass2
        from .pass3.embed import run_embed

        run_crawl(resume = False)

        chunks = run_pass2(only_changed= True, output_path=args.output)
        print(f"Produced {len(chunks)} chunks → {args.output}")

        n = run_embed(resume=True)
        print(f"Upserted {n} vectors to Pinecone")

    elif args.command == 'audit':
        _run_audit()

    else:
        parser.print_help()


def _run_audit():
    """Print audit queries against the manifest DB."""
    from .pass1.db import DB
    db = DB()

    print("\n=== Classification & Strategy Breakdown ===")
    rows = db.conn.execute("""
        SELECT page_classification, chunking_strategy, content_type,
               COUNT(*) as count, ROUND(AVG(word_count)) as avg_words,
               MAX(word_count) as max_words
        FROM pages WHERE crawl_status = 'crawled'
        GROUP BY page_classification, chunking_strategy, content_type
    """).fetchall()
    for r in rows:
        print(f"  {r['page_classification']:15s} | {r['chunking_strategy']:25s} | "
              f"{r['content_type']:5s} | n={r['count']:4d} | "
              f"avg={r['avg_words'] or 0:.0f}w | max={r['max_words'] or 0}w")

    print("\n=== Exclusion Audit ===")
    rows = db.conn.execute("""
        SELECT exclusion_reason, COUNT(*) as count
        FROM pages WHERE crawl_status = 'excluded'
        GROUP BY exclusion_reason ORDER BY count DESC
    """).fetchall()
    for r in rows:
        print(f"  {r['count']:4d}  {r['exclusion_reason']}")

    print("\n=== Depth Distribution ===")
    rows = db.conn.execute("""
        SELECT depth, COUNT(*) as pages FROM pages GROUP BY depth ORDER BY depth
    """).fetchall()
    for r in rows:
        print(f"  depth {r['depth']}: {r['pages']} pages")

    print("\n=== Failed Pages ===")
    rows = db.conn.execute("""
        SELECT url, http_status FROM pages WHERE crawl_status = 'failed'
    """).fetchall()
    for r in rows:
        print(f"  [{r['http_status']}] {r['url']}")

    print("\n=== Duplicate Content ===")
    rows = db.get_duplicate_hashes()
    for r in rows:
        print(f"  {r['dupes']} copies: {r['urls'][:120]}...")

    print("\n=== Top Documents by Inbound Links ===")
    rows = db.conn.execute("""
        SELECT p.url, p.content_type, p.file_size_bytes, COUNT(l.source_url) as inbound
        FROM pages p JOIN links l ON l.target_url = p.url
        WHERE p.content_type != 'html'
        GROUP BY p.url ORDER BY inbound DESC LIMIT 15
    """).fetchall()
    for r in rows:
        size = f"{(r['file_size_bytes'] or 0) / 1024:.0f}KB"
        print(f"  {r['inbound']:3d} links → [{r['content_type']}] {size:>8s}  {r['url']}")

    print("\n=== Leak Check (excluded patterns that snuck through) ===")
    rows = db.conn.execute("""
        SELECT url FROM pages WHERE crawl_status = 'crawled'
        AND (url LIKE '%/elections/20%' OR url LIKE '%/campaign_finance/%'
             OR url LIKE '%/petitions/%' OR url LIKE '%/election_data/%')
    """).fetchall()
    if rows:
        for r in rows:
            print(f"  LEAK: {r['url']}")
    else:
        print("  No leaks detected.")

    print("\n=== Largest Semantic Chunking Candidates ===")
    rows = db.conn.execute("""
        SELECT url, title, word_count FROM pages
        WHERE chunking_strategy = 'semantic_with_overlap'
        ORDER BY word_count DESC LIMIT 15
    """).fetchall()
    for r in rows:
        print(f"  {r['word_count'] or 0:6d}w  {r['title'] or 'Untitled'} — {r['url']}")

    db.close()


if __name__ == '__main__':
    main()

"""
All SQLite read/write operations for the crawl manifest.
Schema is embedded — no external SQL file needed.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    url                     TEXT UNIQUE NOT NULL,
    parent_url              TEXT,
    title                   TEXT,
    section_hierarchy       TEXT,           -- JSON array e.g. ["Voting", "Register", "FAQ"]
    content_type            TEXT,           -- 'html', 'pdf', 'docx', 'xls', 'xlsx', 'csv'
    page_classification     TEXT,           -- 'faq', 'prose', 'table_data', 'form', 'press_release', 'short_static', 'document'
    chunking_strategy       TEXT,           -- 'qa_pairs', 'semantic_with_overlap', 'simple_split', 'ingest_as_single', 'document_extraction', 'skip'
    classification_confidence TEXT,         -- 'high', 'medium', 'low'
    word_count              INTEGER,
    depth                   INTEGER,
    crawl_status            TEXT DEFAULT 'pending',  -- 'pending', 'crawled', 'failed', 'skipped', 'excluded'
    exclusion_reason        TEXT,
    http_status             INTEGER,
    content_hash            TEXT,           -- SHA256 of extracted text, for dedup
    previous_content_hash   TEXT,           -- hash from prior crawl run, for change detection
    file_size_bytes         INTEGER,        -- populated for PDFs/DOCX from HEAD request
    needs_ocr               INTEGER DEFAULT 0,  -- 1 if PDF appears to be image-only
    extracted_snippet       TEXT,           -- first 500 chars of extracted text
    links_out_count         INTEGER,
    discovered_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    crawled_at              TIMESTAMP
);

CREATE TABLE IF NOT EXISTS links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url      TEXT NOT NULL,
    target_url      TEXT NOT NULL,
    link_text       TEXT,
    link_context    TEXT,               -- surrounding sentence, up to 200 chars
    is_internal     INTEGER,            -- 1 = same domain
    is_document     INTEGER,            -- 1 = pdf/docx/xls etc
    discovered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_url, target_url)
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TIMESTAMP,
    completed_at        TIMESTAMP,
    seed_url            TEXT,
    total_discovered    INTEGER,
    total_crawled       INTEGER,
    total_failed        INTEGER,
    total_excluded      INTEGER,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_crawl_status ON pages(crawl_status);
CREATE INDEX IF NOT EXISTS idx_pages_content_hash ON pages(content_hash);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_url);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_url);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DB:
    def __init__(self, db_path: str | None = None):
        path = db_path or DB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- Page operations ----

    def url_exists(self, url: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM pages WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def add_page(self, url: str, parent_url: str | None, depth: int):
        self.conn.execute(
            "INSERT OR IGNORE INTO pages (url, parent_url, depth) VALUES (?, ?, ?)",
            (url, parent_url, depth)
        )
        self.conn.commit()

    def update_page(self, url: str, result: dict, classification: dict):
        self._update_page_row(url, result, classification)
        self.conn.commit()

    def _update_page_row(self, url: str, result: dict, classification: dict):
        """Execute the crawled-page UPDATE without committing — shared by
        update_page() and the atomic update_page_with_children().

        content_hash uses COALESCE so a None incoming value preserves the
        stored hash: document rows are HEAD-only in Pass 1 (extractor hash
        is None), and a plain '= ?' would wipe the hash Pass 2 stored via
        update_document_hash() on every recrawl."""
        self.conn.execute("""
            UPDATE pages SET
                title = ?,
                section_hierarchy = ?,
                content_type = ?,
                page_classification = ?,
                chunking_strategy = ?,
                classification_confidence = ?,
                word_count = ?,
                http_status = ?,
                content_hash = COALESCE(?, content_hash),
                file_size_bytes = ?,
                needs_ocr = ?,
                extracted_snippet = ?,
                links_out_count = ?,
                crawl_status = 'crawled',
                crawled_at = ?
            WHERE url = ?
        """, (
            result.get('title'),
            json.dumps(result.get('section_hierarchy', [])),
            classification.get('content_type'),
            classification.get('page_classification'),
            classification.get('chunking_strategy'),
            classification.get('classification_confidence', 'low'),
            result.get('word_count'),
            result.get('http_status'),
            result.get('content_hash'),
            result.get('file_size_bytes'),
            int(result.get('needs_ocr', False)),
            result.get('snippet'),
            len(result.get('links', [])),
            _utcnow(),
            url
        ))

    def update_page_with_children(self, url: str, result: dict,
                                  classification: dict,
                                  links: list[tuple],
                                  children: list[tuple]):
        """Persist a crawled page, its outbound link rows, and its newly
        discovered child 'pending' rows in ONE transaction.

        Atomicity matters for crash-safe resume: if the parent's 'crawled'
        update committed before the children's pending rows, a crash between
        the two would orphan the children — the parent is already 'crawled'
        so it is never re-processed, and the children were never enqueued.

        Args:
            links:    (source_url, target_url, link_text, link_context,
                       is_internal, is_document) tuples for the links table.
            children: (url, parent_url, depth) tuples for new pending rows.
        """
        try:
            self._update_page_row(url, result, classification)
            self.conn.executemany("""
                INSERT OR IGNORE INTO links
                (source_url, target_url, link_text, link_context, is_internal, is_document)
                VALUES (?, ?, ?, ?, ?, ?)
            """, links)
            self.conn.executemany(
                "INSERT OR IGNORE INTO pages (url, parent_url, depth) VALUES (?, ?, ?)",
                children
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def update_document_hash(self, url: str, content_hash: str) -> None:
        """Set pages.content_hash for a document row.

        Contract with Pass 2: Pass 1 only HEADs documents (PDF/DOCX/XLS),
        so their rows are created with content_hash NULL. Pass 2 calls this
        after fetching the document bytes, storing their SHA256 so that
        snapshot_hashes_for_recrawl() / get_changed_pages() change
        detection works for documents exactly as it does for HTML pages.
        """
        self.conn.execute(
            "UPDATE pages SET content_hash = ? WHERE url = ?",
            (content_hash, url)
        )
        self.conn.commit()

    def get_content_hash(self, url: str) -> str | None:
        """Return pages.content_hash for a URL, or None if unset/unknown."""
        row = self.conn.execute(
            "SELECT content_hash FROM pages WHERE url = ?", (url,)
        ).fetchone()
        return row['content_hash'] if row else None

    def update_status(self, url: str, status: str, reason: str | None = None):
        self.conn.execute(
            "UPDATE pages SET crawl_status = ?, exclusion_reason = ? WHERE url = ?",
            (status, reason, url)
        )
        self.conn.commit()

    def get_page_status(self, url: str) -> str | None:
        """Return the crawl_status for a URL, or None if the URL is unknown."""
        row = self.conn.execute(
            "SELECT crawl_status FROM pages WHERE url = ?", (url,)
        ).fetchone()
        return row['crawl_status'] if row else None

    def get_pending(self) -> list:
        return self.conn.execute(
            "SELECT url, parent_url, depth FROM pages WHERE crawl_status = 'pending' ORDER BY depth, id"
        ).fetchall()

    def get_all_visited_urls(self) -> set[str]:
        """Return URLs that are not pending — used for resume."""
        rows = self.conn.execute(
            "SELECT url FROM pages WHERE crawl_status != 'pending'"
        ).fetchall()
        return {row['url'] for row in rows}

    def snapshot_hashes_for_recrawl(self, urls=None):
        """Copy current content_hash to previous_content_hash.

        Called AFTER a successful full chunk pass (end of `all`), so
        previous_content_hash means "the last content reflected in
        chunks.jsonl". get_changed_pages() then reports pages whose content
        is new or changed since it was last chunked — the baseline for the
        standalone `pass2 --changed` incremental path. Snapshotting before
        the crawl would swallow change events if a run was interrupted
        between crawl and chunking.

        Copies every row that has a hash — including document rows once
        Pass 2 has populated theirs via update_document_hash(). Rows still
        NULL (documents never fetched by Pass 2) keep previous_content_hash
        NULL, so get_changed_pages() keeps returning them until chunked.

        When `urls` (an iterable) is given, only those URLs are snapshotted:
        the caller passes the pages whose state IS reflected in the output
        file — pages that produced chunks plus deliberate-zero pages (see
        run_pass2) — so a page that FAILED stays un-snapshotted and is
        retried by `pass2 --changed`.
        """
        if urls is None:
            self.conn.execute("""
                UPDATE pages SET previous_content_hash = content_hash
                WHERE content_hash IS NOT NULL
            """)
        else:
            self.conn.executemany("""
                UPDATE pages SET previous_content_hash = content_hash
                WHERE url = ? AND content_hash IS NOT NULL
            """, [(u,) for u in urls])
        self.conn.commit()

    # ---- Link operations ----

    def add_link(self, source: str, target: str, text: str,
                 context: str, is_internal: int, is_document: int):
        self.conn.execute("""
            INSERT OR IGNORE INTO links
            (source_url, target_url, link_text, link_context, is_internal, is_document)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source, target, text, context, is_internal, is_document))
        self.conn.commit()

    # ---- Crawl run operations ----

    def start_run(self, seed_url: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawl_runs (started_at, seed_url) VALUES (?, ?)",
            (_utcnow(), seed_url)
        )
        self.conn.commit()
        return cur.lastrowid

    def finalize_run(self, run_id: int):
        stats = self.conn.execute("""
            SELECT
                SUM(CASE WHEN crawl_status = 'crawled'  THEN 1 ELSE 0 END) AS crawled,
                SUM(CASE WHEN crawl_status = 'failed'   THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN crawl_status = 'excluded'  THEN 1 ELSE 0 END) AS excluded,
                COUNT(*) AS total
            FROM pages
        """).fetchone()
        self.conn.execute("""
            UPDATE crawl_runs SET
                completed_at = ?,
                total_crawled = ?,
                total_failed = ?,
                total_excluded = ?,
                total_discovered = ?
            WHERE id = ?
        """, (
            _utcnow(),
            stats['crawled'] or 0,
            stats['failed'] or 0,
            stats['excluded'] or 0,
            stats['total'] or 0,
            run_id
        ))
        self.conn.commit()

    # ---- Query helpers for Pass 2 ----

    def get_crawled_pages(self) -> list:
        return self.conn.execute(
            "SELECT * FROM pages WHERE crawl_status = 'crawled' ORDER BY id"
        ).fetchall()

    def get_changed_pages(self) -> list:
        """Pages `pass2 --changed` must (re)examine.

        HTML pages are returned when their content is new or has changed
        since it was last reflected in chunks.jsonl (i.e. since the last
        snapshot):

        changed = previous_content_hash IS NULL
                  OR (content_hash IS NOT NULL
                      AND content_hash != previous_content_hash)

        Document rows (content_type != 'html', crawled, non-skip strategy —
        the same filters as get_expected_chunk_urls()) are ALWAYS returned:
        Pass 1 only HEADs documents, so the manifest's own hash is whatever
        Pass 2 stored on the last fetch — the manifest cannot detect a
        server-side document change by itself. Pass 2 revalidates each
        returned document cheaply (conditional GET, 304 for unchanged) and
        skips re-chunking when the fetched bytes' sha256 still equals
        previous_content_hash. Rows include previous_content_hash
        (SELECT *) so Pass 2 can make that comparison."""
        return self.conn.execute("""
            SELECT * FROM pages
            WHERE crawl_status = 'crawled'
              AND (
                previous_content_hash IS NULL
                OR (content_hash IS NOT NULL
                    AND content_hash != previous_content_hash)
                OR (content_type IS NOT NULL AND content_type != 'html'
                    AND (chunking_strategy IS NULL OR chunking_strategy != 'skip'))
              )
        """).fetchall()

    def get_pages_by_urls(self, urls) -> list:
        """Full page rows for specific URLs (crawled rows only), in
        manifest order. Used by the --changed merge to pull in dedup
        partners of reprocessed pages."""
        urls = list(urls)
        if not urls:
            return []
        qmarks = ','.join('?' * len(urls))
        return self.conn.execute(f"""
            SELECT * FROM pages
            WHERE crawl_status = 'crawled' AND url IN ({qmarks})
            ORDER BY id
        """, urls).fetchall()

    def clear_snapshot(self, urls) -> None:
        """Force pages back into the --changed set: NULL their
        previous_content_hash so get_changed_pages() keeps returning them
        until they are successfully re-chunked and re-snapshotted. Used for
        pages whose reprocessing failed mid `pass2 --changed` run — their
        chunks.jsonl state can no longer be trusted to match."""
        self.conn.executemany(
            "UPDATE pages SET previous_content_hash = NULL WHERE url = ?",
            [(u,) for u in urls]
        )
        self.conn.commit()

    def get_expected_chunk_urls(self) -> set[str]:
        """URLs of crawled pages that Pass 2 should produce chunks for:
        everything crawled except pages deliberately assigned the 'skip'
        strategy. Used by the post-pass2 coverage report."""
        rows = self.conn.execute("""
            SELECT url FROM pages
            WHERE crawl_status = 'crawled'
              AND (chunking_strategy IS NULL OR chunking_strategy != 'skip')
        """).fetchall()
        return {row['url'] for row in rows}

    def get_duplicate_hashes(self) -> list:
        return self.conn.execute("""
            SELECT content_hash, COUNT(*) as dupes, GROUP_CONCAT(url) as urls
            FROM pages
            WHERE crawl_status = 'crawled' AND content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING dupes > 1
        """).fetchall()

    def close(self):
        self.conn.close()

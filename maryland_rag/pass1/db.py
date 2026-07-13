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
    content_type            TEXT,           -- 'html', 'pdf', 'docx', 'xls', 'csv'
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
                content_hash = ?,
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
        self.conn.commit()

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

    def snapshot_hashes_for_recrawl(self):
        """Before a re-crawl, copy current content_hash to previous_content_hash."""
        self.conn.execute("""
            UPDATE pages SET previous_content_hash = content_hash
            WHERE content_hash IS NOT NULL
        """)
        self.conn.commit()
    
    def is_first_run(self) -> bool:
        """Return True if no pages have been crawled yet (no content hashes exist)."""
        row = self.conn.execute(
            "SELECT 1 FROM pages WHERE content_hash IS NOT NULL LIMIT 1"
        ).fetchone()
        return row is None

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
        """Pages that are new or whose content changed since last crawl."""
        return self.conn.execute("""
            SELECT * FROM pages
            WHERE crawl_status = 'crawled'
              AND (
                previous_content_hash IS NULL
                OR content_hash != previous_content_hash
              )
        """).fetchall()

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

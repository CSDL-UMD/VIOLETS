from maryland_rag.pass1.db import DB
from maryland_rag.pass2.pdf_triage import triage_pdf

# Test the scoring logic directly (no DB, no network)
cases = [
  # (url, file_size_bytes, needs_ocr, expected_bucket)
  ("https://elections.maryland.gov/elections/results/2024_general.pdf", 1_000_000, 0, "process"),
  ("https://elections.maryland.gov/about/meeting_materials/agenda_jan.pdf", 500_000, 0, "skip"),
  ("https://elections.maryland.gov/voting/documents/voter_guide.pdf", 25_000_000, 1, "review"),  
  #large OCR
  ("https://elections.maryland.gov/about/documents/misc.pdf", 800_000, 0, "review"),  # neutral
]

for url, size, ocr, expected in cases:
  result = triage_pdf(url, size, ocr)
  status = "OK" if result['triage_bucket'] == expected else "FAIL"
  print(f"[{status}] {result['triage_bucket']:7s} | {result['reason']}")
  print(f"       URL: {url.split('/')[-1]}")
  print()
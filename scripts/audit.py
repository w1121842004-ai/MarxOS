from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common_cli import print_command_list, python_command


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

AUDIT_COMMANDS = {
    "cache-page-sequence": ("audit_cache_page_sequence.py", "check OCR cache page ordering and gaps"),
    "concept-metadata": ("audit_concept_metadata.py", "inspect concept-title metadata enrichment"),
    "exact-quote-top1": ("audit_exact_quote_top1.py", "sample exact-quote retrieval top1 quality"),
    "ocr-printed-pages": ("audit_ocr_printed_pages.py", "audit printed-page OCR extraction"),
    "page-candidates": ("audit_page_candidates.py", "inspect candidate citation pages for a query"),
    "page-metadata": ("audit_page_metadata.py", "print normalized page metadata summaries"),
    "paragraph-cache": ("audit_paragraph_cache.py", "inspect paragraph cache output quality"),
    "validate-maps": ("validate_maps.py", "validate page_map and article_map JSON structure"),
}


def usage(exit_code: int = 1) -> int:
    print("Usage: python scripts/audit.py <command> [args...]")
    print()
    print_command_list(
        "Available commands:",
        {name: description for name, (_script, description) in AUDIT_COMMANDS.items()},
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help", "help", "list"}:
        return usage(0 if argv else 1)

    command = argv[0]
    command_info = AUDIT_COMMANDS.get(command)
    if not command_info:
        print(f"Unknown audit command: {command}", file=sys.stderr)
        print("", file=sys.stderr)
        return usage()

    script_name, _description = command_info
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        python_command(script_path, *argv[1:]),
        cwd=str(ROOT_DIR),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

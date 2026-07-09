#!/usr/bin/env python3
"""Step 6D.1b: validate a directory of manually browser-saved UFCStats HTML.

Offline and read-only: it does NOT contact the network and does NOT open
data/ufc.db at all. It classifies each cached .html file by content (completed-
events / upcoming-events / event-detail / fighter-detail / browser-challenge /
unknown), previews parsed counts, flags challenge pages, warns if key pages are
missing, and tells you exactly which pages to save next.

UFCStats serves a JavaScript browser-challenge to plain HTTP clients; the
project intentionally does not bypass it. The supported workflow is to save the
pages from a normal browser (File > Save Page As, "Web Page, HTML only") into a
cache directory and run the audit with --offline-cache-only.

Example:
  python scripts/validate_step6d1_cache.py --cache-dir data/cache/ufcstats
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from ufc_pipeline.step6d_ufcstats_update_audit import validate_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="data/cache/ufcstats")
    p.add_argument("--output-dir", default=None, help="Optional: write a validation report here (json+md).")
    args = p.parse_args()

    result = validate_cache(args.cache_dir)
    print(f"Cache validation for: {args.cache_dir}")
    print(f"  exists: {result['cache_dir_exists']} | .html files: {result['n_html_files']}")
    print(f"  page types: {result['page_types_detected']}")
    print(f"  has completed-events page: {result['has_completed_events_page']} | "
          f"has event-detail page: {result['has_event_detail_page']} | "
          f"challenge pages: {len(result['challenge_pages_detected'])}")
    for f in result["files"]:
        extra = ""
        if "events_parsed" in f:
            extra = f" -> {f['events_parsed']} events"
        elif "fights_parsed" in f:
            extra = f" -> {f['fights_parsed']} fights, {f['fighters_parsed']} fighters ({f.get('event_title')})"
        print(f"    [{f['page_type']}] {f['file']} ({f['bytes']} bytes){extra}")
    if result["warnings"]:
        print("  warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")
    print(f"  ready for offline audit: {result['ready_for_offline_audit']}")
    print(f"  -> {result['recommendation']}")
    if not result["ready_for_offline_audit"]:
        print("  Save these pages (any of these filenames works):")
        for rec in result["recommended_files_to_save"]:
            print(f"    - {rec['filename']}: {rec['description']}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "step6d1_cache_validation.json").write_text(json.dumps(result, indent=2))
        (out / "step6d1_cache_validation.md").write_text(_render_md(result))
        print(f"  report: {out}/step6d1_cache_validation.{{json,md}}")


def _render_md(result: dict) -> str:
    lines = ["# Step 6D.1b: UFCStats Cache Validation", "",
             f"Cache dir: `{result['cache_dir']}` (exists: {result['cache_dir_exists']}, "
             f"{result['n_html_files']} .html files)", "",
             f"Ready for offline audit: **{result['ready_for_offline_audit']}**", "",
             f"> {result['recommendation']}", "", "## Files", ""]
    lines.append("| file | type | bytes | parsed |")
    lines.append("|---|---|---:|---|")
    for f in result["files"]:
        parsed = (f"{f['events_parsed']} events" if "events_parsed" in f else
                  f"{f.get('fights_parsed', '')} fights / {f.get('fighters_parsed', '')} fighters" if "fights_parsed" in f else "")
        lines.append(f"| {f['file']} | {f['page_type']} | {f['bytes']} | {parsed} |")
    lines.append("")
    if result["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in result["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Recommended Files To Save")
    lines.append("")
    for rec in result["recommended_files_to_save"]:
        lines.append(f"- `{rec['filename']}`: {rec['description']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

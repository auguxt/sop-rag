"""Command-line interface for the Manufacturing Quality SOP Assistant.

Examples
--------
    python cli.py "What are the steps for incoming material inspection?"
    python cli.py "What is different between incoming and final inspection?" --sources
    python cli.py --interactive
    python cli.py --list-docs
    python cli.py --rebuild --embed local "How often must vernier calipers be calibrated?"
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import textwrap


def _plain(markdown: str) -> str:
    """Make the markdown answer readable in a terminal."""
    text = markdown.replace("**", "").replace("*", "")
    return re.sub(r"\n{3,}", "\n\n", text)


def print_comparison(comp) -> None:
    """Two clearly labelled columns per aspect (terminal friendly)."""
    width = 43
    a_head, b_head = f"{comp['title_a']} ({comp['doc_a']})", f"{comp['title_b']} ({comp['doc_b']})"
    print(f"COMPARISON: {a_head}  vs  {b_head}")
    print("Every line is copied from the cited SOP; 'Not specified' = section absent in that document.")
    if comp["key_differences"]:
        print("\nKey documented differences (numeric):")
        for d in comp["key_differences"]:
            print(f"  - {d['aspect']} / {d['quantity']}: {d['a_value']} {d['a_citation']}  vs  {d['b_value']} {d['b_citation']}")
    for row in comp["rows"]:
        print("\n" + "-" * 90)
        print(f"{row['aspect'].upper()}  [{row['status']}]")
        print(f"{comp['title_a'][:width]:<{width}} | {comp['title_b'][:width]}")
        left = [l for item in row["a_items"] for l in textwrap.wrap("- " + item, width)] + [row["a_citation"]]
        right = [l for item in row["b_items"] for l in textwrap.wrap("- " + item, width)] + [row["b_citation"]]
        for i in range(max(len(left), len(right))):
            l = left[i] if i < len(left) else ""
            r = right[i] if i < len(right) else ""
            print(f"{l:<{width}} | {r}")


def print_answer(ans, show_sources: bool) -> None:
    print("\n" + "=" * 90)
    print(f"Question      : {ans.question}")
    print(f"Question type : {ans.question_type}   Route: {ans.route}")
    print(f"Answer status : {ans.status}   Time: {ans.processing_ms:.0f} ms   Generator: {ans.generator}")
    if ans.warnings:
        print(f"Warnings      : {'; '.join(ans.warnings)}")
    print("-" * 90)
    if ans.comparison:
        print_comparison(ans.comparison)
        print("\n" + _plain(ans.answer_markdown.split("---")[-1]).strip())
    else:
        print(_plain(ans.answer_markdown))
    if show_sources and ans.sources:
        print("-" * 90)
        print("SOURCES (supporting passages):")
        for i, s in enumerate(ans.sources, start=1):
            page = f", page {s['page']}" if s["page"] not in ("n/a", "None") else ""
            print(f"\n[{i}] {s['doc_id']} - {s['title']} | {s['section']}{page} | file: {s['file_name']} | "
                  f"role: {s['role']} | score: {s['score']}")
            print(textwrap.indent(s["text"], "     "))
    print("=" * 90)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manufacturing Quality SOP Assistant (LlamaIndex RAG)")
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="interactive question loop")
    parser.add_argument("-s", "--sources", action="store_true", help="print the supporting passages")
    parser.add_argument("--rebuild", action="store_true", help="re-ingest the documents and rebuild the index")
    parser.add_argument("--embed", choices=["auto", "huggingface", "local"], help="embedding back-end")
    parser.add_argument("--list-docs", action="store_true", help="list the indexed documents")
    parser.add_argument("-v", "--verbose", action="store_true", help="show info logs")
    args = parser.parse_args(argv)

    try:  # avoid UnicodeEncodeError for characters such as '§' on old Windows consoles
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if args.embed:
        os.environ["SOP_EMBED_BACKEND"] = args.embed

    from sop_assistant.assistant import SOPAssistant

    print("Loading documents and index ...")
    assistant = SOPAssistant(rebuild=args.rebuild)
    info = "loaded from storage/" if assistant.index_info["loaded_from_storage"] else "built and saved to storage/"
    print(f"Ready: {len(assistant.documents)} documents, {len(assistant.nodes)} section nodes, "
          f"embeddings={assistant.embed_name}, LLM={assistant.llm_name} (index {info})")

    if args.list_docs:
        for d in assistant.documents:
            print(f"  {d['doc_id']:<13} {d['file_type']:<4} rev {d['revision']:<2} {d['sections']:>2} sections  {d['title']}")

    if args.question:
        print_answer(assistant.ask(" ".join(args.question)), args.sources)
    if args.interactive:
        print("\nType a question (or 'exit'). Add ' -s' at the end to show sources.")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            show = args.sources
            if q.endswith(" -s"):
                q, show = q[:-3], True
            if q:
                print_answer(assistant.ask(q), show)
    if not (args.question or args.interactive or args.list_docs):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Requirement 01 - document collection, ingestion and indexing.

Pipeline
--------
1. ``load_documents``  : read every PDF (page by page, with pypdf) and every
   .txt/.md file in the data folder into LlamaIndex ``Document`` objects.
   Page numbers are kept for PDFs.
2. ``parse_sections``  : rebuild logical lines (PDF text wraps lines), read
   the document header (Document ID, Title, Revision ...) and split the body
   into numbered sections ("5. Inspection Steps").
3. ``build_nodes``     : one LlamaIndex ``TextNode`` per section. Metadata
   preserves document identity (doc_id, title, file) and the section label
   and page range, so every answer can be cited precisely.
4. ``build_or_load_index`` : embed the nodes and store them in a
   ``VectorStoreIndex``; the index is persisted in ``storage/`` and re-used
   while the documents and the embedding model are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

log = logging.getLogger(__name__)

PARSER_VERSION = "3"
SUPPORTED_EXTS = (".pdf", ".txt", ".md")

HEADER_RE = re.compile(r"^(Document ID|Title|Document Type|Revision|Effective Date|Owner):\s*(.+)$")
HEADING_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^.:]{1,70})$")
ITEM_START_RE = re.compile(r"^(-\s+|\d{1,2}\.\d{1,2}\s+[A-Z])")
PDF_NOISE_RE = re.compile(r"^(Page \d+|.*Controlled copy)$")
DEFECT_CODE_RE = re.compile(r"\b(SC|DM|LB|CF|DN|CR|BR)-0\d\b")
DOC_REF_RE = re.compile(r"\b(?:SOP|REF|CHK)-[A-Z]{2,4}-\d{3}\b")

# Metadata that is useful for citations but should not influence the embedding.
_NOT_EMBEDDED = ["file_name", "file_type", "page_label", "section_number", "defect_codes",
                 "references", "revision", "doc_type", "effective_date", "owner"]


@dataclass
class Section:
    number: int
    title: str
    lines: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.number}. {self.title}"

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "n/a"
        lo, hi = min(self.pages), max(self.pages)
        return str(lo) if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def _read_pdf(path: Path) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_documents(data_dir: Path) -> List[Document]:
    """Load PDF and text files. PDFs give one Document per page (with ``page``)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Document folder not found: {data_dir}")
    docs: List[Document] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.suffix.lower() == ".pdf":
            for page_no, text in _read_pdf(path):
                docs.append(Document(text=text, metadata={"file_name": path.name, "file_type": "pdf", "page": page_no}))
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(text=text, metadata={"file_name": path.name, "file_type": path.suffix.lstrip(".").lower(), "page": None}))
    if not docs:
        raise ValueError(f"No PDF/TXT/MD documents found in {data_dir}")
    return docs


# ---------------------------------------------------------------------------
# 2. Section parsing
# ---------------------------------------------------------------------------
def _logical_lines(pages: List[Tuple[Optional[int], str]]) -> List[Tuple[Optional[int], str]]:
    """Join wrapped PDF lines back into logical lines (list items / paragraphs)."""
    out: List[Tuple[Optional[int], str]] = []
    start_new = True
    for page, text in pages:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                start_new = True
                continue
            if PDF_NOISE_RE.match(line):
                continue
            is_start = bool(HEADER_RE.match(line) or HEADING_RE.match(line) or ITEM_START_RE.match(line))
            if start_new or is_start or not out:
                out.append((page, line))
            else:
                prev_page, prev = out[-1]
                out[-1] = (prev_page, f"{prev} {line}")
            # a heading is always followed by a new logical line
            start_new = bool(HEADING_RE.match(line))
    return out


def parse_sections(pages: List[Tuple[Optional[int], str]]) -> Tuple[Dict[str, str], List[Section]]:
    header: Dict[str, str] = {}
    sections: List[Section] = []
    for page, line in _logical_lines(pages):
        m_head = HEADER_RE.match(line)
        m_sec = HEADING_RE.match(line)
        if m_head and not sections:
            header[m_head.group(1)] = m_head.group(2).strip()
        elif m_sec:
            sections.append(Section(number=int(m_sec.group(1)), title=m_sec.group(2).strip()))
            if page is not None:
                sections[-1].pages.append(page)
        elif sections:
            sections[-1].lines.append(line)
            if page is not None:
                sections[-1].pages.append(page)
        else:
            header.setdefault("Preamble", "")
            header["Preamble"] = (header["Preamble"] + " " + line).strip()
    return header, sections


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def build_nodes(page_docs: List[Document]) -> List[TextNode]:
    """Group page documents by file, parse sections and create one node per section."""
    by_file: Dict[str, List[Document]] = {}
    for d in page_docs:
        by_file.setdefault(d.metadata["file_name"], []).append(d)

    splitter = SentenceSplitter(chunk_size=700, chunk_overlap=60)
    nodes: List[TextNode] = []
    for file_name, pages in by_file.items():
        pages.sort(key=lambda d: d.metadata.get("page") or 0)
        header, sections = parse_sections([(d.metadata.get("page"), d.text) for d in pages])
        doc_id = header.get("Document ID") or file_name.split("_")[0]
        title = header.get("Title") or Path(file_name).stem
        parent = Document(
            text="\n".join(d.text for d in pages),
            doc_id=doc_id,
            metadata={"doc_id": doc_id, "title": title, "file_name": file_name},
        )
        if not sections:  # document without numbered headings -> one section
            sections = [Section(number=1, title="Full Document", lines=[parent.text])]
        for sec in sections:
            body = "\n".join(sec.lines).strip()
            if not body:
                continue
            chunks = splitter.split_text(body) if len(body) > 2500 else [body]
            for part, chunk in enumerate(chunks, start=1):
                node_id = f"{doc_id}::S{sec.number}" + (f"::P{part}" if len(chunks) > 1 else "")
                node = TextNode(
                    id_=node_id,
                    text=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "section": sec.label,
                        "section_number": sec.number,
                        "page_label": sec.page_label,
                        "file_name": file_name,
                        "file_type": pages[0].metadata.get("file_type", ""),
                        "doc_type": header.get("Document Type", ""),
                        "revision": header.get("Revision", ""),
                        "effective_date": header.get("Effective Date", ""),
                        "owner": header.get("Owner", ""),
                        "defect_codes": ",".join(sorted({m.group(0) for m in DEFECT_CODE_RE.finditer(chunk)})),
                        "references": ",".join(sorted({r for r in DOC_REF_RE.findall(chunk) if r != doc_id})),
                    },
                    excluded_embed_metadata_keys=list(_NOT_EMBEDDED),
                    excluded_llm_metadata_keys=list(_NOT_EMBEDDED),
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=parent.doc_id)
                nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# 4. Index
# ---------------------------------------------------------------------------
def _fingerprint(data_dir: Path, embed_name: str) -> str:
    h = hashlib.md5()
    h.update(PARSER_VERSION.encode())
    h.update(embed_name.encode())
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTS:
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def build_or_load_index(data_dir: Path, storage_dir: Path, embed_model, embed_name: str,
                        persist: bool = True, rebuild: bool = False):
    """Return ``(index, nodes, info_dict)``."""
    data_dir, storage_dir = Path(data_dir), Path(storage_dir)
    fp = _fingerprint(data_dir, embed_name)
    fp_file = storage_dir / "fingerprint.json"

    if persist and not rebuild and fp_file.exists():
        try:
            saved = json.loads(fp_file.read_text())
            if saved.get("fingerprint") == fp:
                ctx = StorageContext.from_defaults(persist_dir=str(storage_dir))
                index = load_index_from_storage(ctx, embed_model=embed_model)
                nodes = sorted(index.docstore.docs.values(), key=lambda n: n.node_id)
                return index, nodes, {"loaded_from_storage": True, "fingerprint": fp}
        except Exception as exc:  # corrupt / incompatible storage -> rebuild
            log.warning("Could not load persisted index (%s); rebuilding.", exc)

    nodes = build_nodes(load_documents(data_dir))
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=False)
    if persist:
        storage_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_dir))
        fp_file.write_text(json.dumps({"fingerprint": fp, "embed_model": embed_name}, indent=2))
    return index, nodes, {"loaded_from_storage": False, "fingerprint": fp}

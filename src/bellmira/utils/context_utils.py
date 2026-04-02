"""
Utilities for building context dicts used by evaluators.

All loaders return ``Dict[str, str]``: a label → text mapping where each
value is one context string.  Pass the result directly to any evaluator that
accepts a ``contexts`` parameter.

Quick reference
---------------
contexts_from_splits      Slice a single text at explicit markers (e.g. chapter headings).
contexts_from_word_counts Truncate a single text at N word boundaries.
contexts_from_files       Load one context per file.
contexts_from_bible       Convenience wrapper for the bundled Bible corpus.
"""

from pathlib import Path
from typing import Dict, List, Optional


def read_text_file(path: str, encoding: str = "latin-1") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def contexts_from_splits(
    text: str,
    split_markers: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    For each marker, take everything in *text* **before** that marker as one context.

    Placing markers at chapter or section headings produces contexts of
    increasing length — useful for context-length sweeps.

    Args:
        text:          Full source text.
        split_markers: List of strings to split on.  Each context is the
                       portion of *text* that precedes the corresponding marker.
        labels:        Optional list of keys for the returned dict.
                       Defaults to ``context_1``, ``context_2``, …

    Returns:
        Ordered dict of ``{label: context_text}``.

    Example::

        contexts = contexts_from_splits(
            text,
            split_markers=["Chapter 5\\n", "Chapter 10\\n", "Chapter 20\\n"],
            labels=["ch5", "ch10", "ch20"],
        )
    """
    if labels and len(labels) != len(split_markers):
        raise ValueError("len(labels) must equal len(split_markers)")

    contexts: Dict[str, str] = {}
    for i, marker in enumerate(split_markers):
        label = labels[i] if labels else f"context_{i + 1}"
        parts = text.split(marker, 1)
        contexts[label] = parts[0] if len(parts) > 1 else text
    return contexts


def contexts_from_word_counts(
    text: str,
    word_counts: List[int],
    labels: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Truncate *text* at each word count to build contexts of increasing length.

    Args:
        text:        Source text.
        word_counts: List of word counts (e.g. ``[500, 1000, 2000, 4000]``).
        labels:      Optional keys.  Defaults to ``500w``, ``1000w``, …

    Returns:
        Ordered dict of ``{label: context_text}``.

    Example::

        contexts = contexts_from_word_counts(
            text,
            word_counts=[500, 1000, 2000, 4000],
        )
    """
    if labels and len(labels) != len(word_counts):
        raise ValueError("len(labels) must equal len(word_counts)")

    words = text.split()
    contexts: Dict[str, str] = {}
    for i, count in enumerate(word_counts):
        label = labels[i] if labels else f"{count}w"
        contexts[label] = " ".join(words[:count])
    return contexts


def contexts_from_files(
    file_paths: List[str],
    labels: Optional[List[str]] = None,
    encoding: str = "utf-8",
) -> Dict[str, str]:
    """
    Load each file as a separate context.  The label defaults to the file stem.

    Args:
        file_paths: List of paths to text files.
        labels:     Optional keys.  Defaults to the stem of each file name.
        encoding:   Text encoding (default ``utf-8``).

    Returns:
        Dict of ``{label: file_contents}``.

    Example::

        contexts = contexts_from_files([
            "docs/product_a.txt",
            "docs/product_b.txt",
        ])
    """
    if labels and len(labels) != len(file_paths):
        raise ValueError("len(labels) must equal len(file_paths)")

    contexts: Dict[str, str] = {}
    for i, path in enumerate(file_paths):
        label = labels[i] if labels else Path(path).stem
        contexts[label] = read_text_file(path, encoding=encoding)
    return contexts


def contexts_from_bible(
    bible_path: str,
    chapter_numbers: List[int],
    book: str = "Genesis",
) -> Dict[str, str]:
    """
    Build contexts from the bundled Bible corpus using chapter boundaries.

    Each context is the portion of text that precedes chapter *n* of *book*,
    producing contexts of increasing length as chapter numbers grow.

    This is a convenience wrapper around :func:`contexts_from_splits` that
    preserves the original evaluator behaviour.

    Args:
        bible_path:      Path to ``resources/data/text/bible.txt``.
        chapter_numbers: Chapter numbers to split at (e.g. ``[2, 5, 9, 14]``).
        book:            Bible book name (default ``"Genesis"``).

    Returns:
        Dict of ``{label: context_text}``.

    Example::

        from bellmira.utils.context_utils import contexts_from_bible

        contexts = contexts_from_bible(
            bible_path="/workspaces/BeLLMira/resources/data/text/bible.txt",
            chapter_numbers=[2, 5, 9, 14, 22],
        )
    """
    text = read_text_file(bible_path)
    markers = [f"{book} {num}:1\t" for num in chapter_numbers]
    labels = [f"{book}_ch{num}" for num in chapter_numbers]
    return contexts_from_splits(text, markers, labels)

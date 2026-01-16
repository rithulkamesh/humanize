Dataset Guide
=============

Humanize uses open, human-written text stored in a columnar format to extract structural writing patterns.
The dataset is designed to be offline-first, append-only, and fully inspectable.

This document explains what data to collect, how to store it, and how much is needed.

Data format
-----------

All datasets are stored using Apache Arrow (or Parquet) with explicit schemas.
There are no databases, services, or background processes involved.

The canonical dataset is an append-only table called ``writing_events``.

Writing events
--------------

Each row represents a single writing event.

Required columns:

- id (string): unique identifier
- text (string): raw human-written text
- source (string): where the text came from (e.g. user_submission)
- license (string): licensing for the text (default: CC0)
- created_at (timestamp): when the event was added

Rules:
- Text must be written by a human
- Do not preprocess or clean the text
- Preserve punctuation, casing, and spacing
- Do not include AI-generated content

Raw data is never modified after being added.

Licensing
---------

All included dataset entries must be licensed under Creative Commons Zero (CC0) unless explicitly stated otherwise.

By contributing text, you confirm that:
- You have the legal right to share it
- It is not proprietary or confidential
- You agree to release it under CC0

The dataset exists to extract structural patterns, not to reproduce content.

Derived datasets
----------------

Derived datasets are generated from ``writing_events`` and may be safely deleted and regenerated.

Examples include:
- sentence-level tables
- structural feature tables
- aggregate style profiles

Derived data must always be reproducible from the raw dataset.

How much data is needed
-----------------------

Humanize does not require large datasets.

Recommended sizes:
- Minimum viable: 30–50 writing events
- Solid baseline: 100–150 writing events
- More than enough: 300 writing events

Structural writing statistics converge quickly.
Quality matters more than quantity.

What to avoid
-------------

- Scraped or copyrighted text
- Long documents or transcripts
- Lists, bullet points, or tables
- Marketing or SEO copy
- AI-generated writing

Removal requests
----------------

Contributors may request removal of their data at any time.
Removal requests will be honored without question.

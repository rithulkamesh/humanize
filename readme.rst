Humanize
========

**An open-source, rule-based system for humanizing text using open datasets and deterministic transformations.**

Humanize is a small, transparent toolkit for improving the naturalness of text that feels overly rigid or machine-like.

It does not generate new content.
It does not depend on hosted AI services.
It does not require an internet connection.

Instead, it works by applying simple, inspectable transformations informed by open, human-written reference data.

Why this exists
---------------

Many tools that claim to “humanize” text rely on large language models running behind closed APIs.
They are expensive, opaque, and difficult to reason about.

Humanize takes a different path.

Human writing has measurable structure:
sentence length varies,
punctuation creates pauses,
perfect symmetry is rare,
and slight irregularity often reads as more natural than polish.

These properties can be modeled and applied deterministically, without guessing or generation.

What Humanize does
------------------

- Accepts a block of text as input
- Analyzes structural features such as sentence length and rhythm
- Compares them against open reference examples of human writing
- Applies rule-based transformations to adjust flow and pacing
- Outputs revised text while preserving the original meaning

All steps are explicit and reproducible.

What Humanize does not do
-------------------------

- It does not generate text from scratch
- It does not scrape or reuse proprietary content
- It does not require API keys or paid services
- It does not train models on user input
- It does not run background processes or telemetry

Offline-first by design
-----------------------

Humanize is designed to run entirely offline.

Reference data is stored as plain text files.
Transformations are rule-based.
The core pipeline is deterministic.

You should be able to clone the repository, disconnect from the internet, and still use the system end to end.

Open datasets
-------------

Style reference data consists of short, human-written text chunks stored as individual files.
All included data is either user-contributed with consent or released under permissive licenses.

The intent is not to reproduce content, but to capture structural patterns common in human writing.

Contributors retain control over their submissions, and data can be removed upon request.

License
-------

Humanize is licensed under the **GNU General Public License v3 (GPLv3)**.

This ensures that all distributed modifications and forks remain open source.
The intent is to keep the project transparent, auditable, and freely usable by the community.

If you build on this project, those improvements should remain available to others as well.

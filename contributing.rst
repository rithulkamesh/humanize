Contributing
============

Thanks for your interest in contributing to Humanize.

This project is intentionally small, transparent, and rule-driven. Contributions are welcome, but the goal is not to grow complexity for its own sake.

Before contributing, please read this document carefully.

Guiding principles
------------------

Humanize follows a few core principles that guide all contributions:

- Offline-first by default
- Deterministic behavior over probabilistic behavior
- Open datasets and inspectable logic
- Minimal reliance on large language models
- No hidden compute or background services

If a change conflicts with these principles, it is unlikely to be accepted.

Licensing and forks
-------------------

Humanize is licensed under the GNU General Public License (GPL).

By contributing, you agree that:
- Your contributions will be licensed under the GPL
- Any distributed forks or modifications must also remain open source under the same license

This is intentional. The project is designed to remain free, auditable, and community-owned.

If this licensing model does not work for your use case, this may not be the right project to contribute to.

What to contribute
------------------

Good contributions include:
- New deterministic text transformation rules
- Improvements to existing rewrite passes
- Bug fixes and test coverage
- Documentation and examples
- Open, properly licensed style reference data

Less helpful contributions include:
- Large architectural rewrites without prior discussion
- Features that require always-on services
- Changes that introduce opaque or non-reproducible behavior

If you are unsure, open an issue before starting work.

Data contributions
------------------

If you contribute text data:
- Only submit text you have the right to share
- Do not submit scraped or proprietary content
- Keep examples short (1–3 sentences)
- Preserve original punctuation and formatting

You may request removal of your contributed data at any time.

Optional LLM integrations
-------------------------

Local language model integrations are allowed, but must follow these rules:
- Disabled by default
- Clearly isolated from the core pipeline
- No network calls unless explicitly configured by the user
- Sentence-level access only

LLMs are considered optional polish, not a dependency.

Code style and scope
--------------------

- Prefer small, readable functions
- Avoid clever abstractions
- Favor clarity over performance unless justified
- One responsibility per file

If a file starts to feel “smart,” it is probably doing too much.

Submitting changes
------------------

- Fork the repository
- Create a focused branch
- Keep commits small and descriptive
- Open a pull request with a clear explanation of what changed and why

Discussion is welcome. Bikeshedding is not.

Thank you
---------

Humanize exists because people care about transparent tools.

If you help make it better, that matters.

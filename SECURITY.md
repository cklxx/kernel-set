# Security Policy

## Supported versions

kernel-set is pre-1.0; only the latest released version on `main` receives
security fixes.

| Version | Supported |
|---------|-----------|
| 0.2.x   | yes       |
| < 0.2   | no        |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately via GitHub's
[private vulnerability reporting](https://github.com/cklxx/kernel-set/security/advisories/new)
("Report a vulnerability" under the repository's Security tab). If that is
unavailable, open a minimal issue asking a maintainer to contact you and we will
arrange a private channel.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected version / commit and GPU arch if relevant.

We aim to acknowledge reports within a few business days and will keep you
informed of remediation progress. Coordinated disclosure is appreciated: please
give us a reasonable window to ship a fix before any public disclosure.

## Scope

This is a GPU kernel library invoked through a C ABI. The most relevant classes
of issue are memory-safety bugs reachable from the public `ks_*` entry points
(out-of-bounds device access, missing shape/pointer validation) and incorrect
numerical results that could affect downstream model integrity. Build-time
supply-chain concerns in vendored `third_party/` sources should be reported to
their upstream projects (see `THIRD_PARTY_NOTICES.md`).

# Final Release Check

Checked on 2026-10-10. Scope: this local package; GitHub upload is not included.

## Numerical Checks

- Python syntax and help commands passed for both entry points and the tests.
- The complete `python reproduce.py all` run exited successfully.
- Valid core combinations: 10,368; profit increases: 4,062; decreases: 6,306.
- All 14 generated CSV/JSON files matched the reference MD5 checksums.
- Core-grid MD5: `2d9d10e6273cfe37a0b36a73498725a3`.
- Five numerical tests passed: grid constraints, reference profits, supply
  cap, reference-file integrity, and verification failure detection.
- Verification also passed when calling the entry point from outside the project.
- Reference CSV/JSON bytes are protected against Git line-ending conversion.

## Publication Presentation

All six figures use English text and Times New Roman. The font is installed
on the validation computer and is checked explicitly before plotting.
Figures were regenerated from unchanged reference tables and inspected for
layout. The physician-efficiency panel uses its own efficiency coordinates,
not the referral-rate coordinates.
README, method documentation, release notes, and dependency notices use English;
the authors' requested Chinese names are retained.

## Cleanup and Preservation

The original batch robustness solver was retained to preserve the batch-wide
fixed-point stopping condition. Old modules and duplicate documentation were
backed up in the system temporary directory `healthcare-legacy-20261010`.
They are not upload content.

Generated output, Python caches, and temporary formatting tools are excluded
from the release folder. Keep `expected_outputs/` as the reproduction baseline.

## Environment and Limits

Windows; Python 3.9.0; NumPy 2.0.2; Matplotlib 3.9.4.
Other operating systems and a fresh Conda installation have not been tested.
Platform-dependent floating-point arithmetic can affect exact matching.
This record establishes numerical regression consistency, not independent
peer review of the paper's theoretical conclusions.

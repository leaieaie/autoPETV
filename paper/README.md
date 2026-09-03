# autoPET V — preprint

LNCS short paper for the autoPET V final submission.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The paper |
| `llncs.cls`, `splncs04.bst` | Springer LNCS class, copied from `../publication_template/autoPET Template.zip` |

## Build

No LaTeX toolchain is installed on the authoring machine. Either build on
Overleaf (upload the three files above, set the compiler to pdfLaTeX), or
locally:

```
pdflatex main && pdflatex main
```

Two passes are needed for the cross-references in Tables 1-3.

## Before submitting

- [x] Author and affiliation filled in (Naoki Haku, School of Health Sciences,
      Faculty of Medicine, Hirosaki University)
- [x] Team name filled in (`25plqki`) in Table 3 (`tab:details`)
- [x] Built and submitted to Grand Challenge with the final submission
      (2026-09-01, `main.pdf`, 7 pages)
- [ ] Add the 10-fold preliminary AUC values to Table 2 — the text currently
      says only that it scored above the 6-fold variant
- [ ] Add an e-mail address to the `\institute` block if one should be published
- [ ] Confirm the LesionLocator citation details (authors, venue) against the
      arXiv record for 2508.21680 before any arXiv submission

## Scope note

The paper reports negative results: six retrained configurations that did not
reach the organizers' baseline, and the measurement mistakes made along the
way. The final submission uses publicly released Apache-2.0 weights from
`MIC-DKFZ/autoPET-interactive`, which the challenge rules permit. The paper
states this explicitly rather than presenting the network as our own.

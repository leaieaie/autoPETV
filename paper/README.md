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

- [ ] Fill in `FIRST AUTHOR`, `AUTHOR@EXAMPLE.AC.JP`, the ORCID iD and the
      affiliation in the `\author` / `\institute` block
- [ ] Fill in `TEAM NAME` in Table 3 (`tab:details`) so it matches the team
      name shown on Grand Challenge
- [ ] Replace the placeholder AUC values for the 10-fold submission once the
      final test phase result is known, or drop that row if it is not published
      before submission
- [ ] Confirm the LesionLocator citation details (authors, venue) against the
      arXiv record for 2508.21680

## Scope note

The paper reports negative results: six retrained configurations that did not
reach the organizers' baseline, and the measurement mistakes made along the
way. The final submission uses publicly released Apache-2.0 weights from
`MIC-DKFZ/autoPET-interactive`, which the challenge rules permit. The paper
states this explicitly rather than presenting the network as our own.

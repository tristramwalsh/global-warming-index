# Published outputs

GWI results as published, one folder per release, with the code that made them:
`IGCC/` (annual IGCC updates), `WMO/` (WMO State of the Global Climate) and
`papers/` (individual papers, `<Author>_et_al-<year>`). The Releases table in the
repository README lists them all.

Everything in this folder is licensed CC BY 4.0 (see `LICENSE.md`).

The recipes and scheduling scripts kept with each release run with the code
that made it, not necessarily with the latest: to rerun one, first check out
the commit its README gives (`git checkout <hash>`). The outputs themselves
are all here, so rerunning is rarely needed.

## README template for a new folder

Copy this into `<folder>/README.md` and fill it in:

```markdown
# <Release name>

GWI outputs for <what they were for>.

## Publication and code

| Type | Name | DOI |
| --- | --- | --- |
| Publication | <Author/s (year)>; or: in review | [<DOI>](https://doi.org/<DOI>) |
| Published dataset | <Name>; or: None | [<DOI>](https://doi.org/<DOI>) |
| GWI code | Tag `<tag>` (commit `<hash>`); Zenodo version `<tag>` of the concept DOI | [10.5281/zenodo.24020859](https://doi.org/10.5281/zenodo.24020859) |

## Contents

| Scenario | Description | Results | Recipe |
| --- | --- | --- | --- |
| `<scenario>` | <one sentence> | ✅ | ✅ `recipes/<file>` |

✅ included · ❌ not included

## License

Outputs: CC BY 4.0. Cite the GWI code version and the publication.
```

One Contents row per scenario run. If the outputs are split into subfolders, add
a Folder column, as in `papers/Thorne_et_al-2026/`.

Write DOIs as plain links with the DOI as the text, not badges: they can be read,
copied and searched anywhere.

A release cannot contain its own Zenodo DOI, which is only assigned when the
release is published. So give the concept DOI and name the version by its tag, as
above: that is known when you commit, and Zenodo names each version after its
tag. If you like, swap in the version's own DOI in a later commit.

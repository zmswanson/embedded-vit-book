Authoritative submitted baseline

These files preserve the manuscript supplied for review. Keep them unchanged;
make revisions in ../bookchapter.tex and the other files in the parent directory.

source.zip is an exact copy of:
  /home/zms/scalable_vits_for_embedded_systems.zip
SHA-256:
  653ec0d6aba5ce05e8c8a3a5cc83a009672063f99768c1caf03268f6fb113547

submitted.pdf is an exact copy of:
  /home/zms/scalable_vits_for_embedded_systems-3.pdf
SHA-256:
  4b64fba63c3281a8ab7f71b9ebbef125e45a456935509d3c572dd666ee7f4795

bookchapter.tex and references.bib are exact extracted copies for convenient
text comparisons. Use source.zip for the complete buildable baseline, including
the class file and figures. manifest.json lists the 38 actual source files in
the ZIP with sizes and SHA-256 hashes; macOS resource-fork entries are excluded.
Its matches_original_repo field compares each file to repository commit
f3b559efd5b27f6a72dece7db0462349015a1b1d, before this revision work.

The source ZIP was rebuilt in isolation with latexmk. Its 37-page output matches
the submitted PDF's normalized words on every page, including the references
(20,417 words in total). Normalization removes whitespace/line-wrap differences
and running page headers, and rejoins words hyphenated across line breaks.
Binary PDF equality is not asserted: the submitted PDF and local rebuild use
different pdfTeX versions and have different build metadata.

The parent directory's bookchapter.tex is the submitted source with the reviewer
and consistency corrections applied. Two other source assets were subsequently
revised: figures/jetson_unified_table.tex omits an inconsistent parameter-count
column, and figures/pruning_method_comparison.pdf corrects the Swin-B baseline
panel's checkpoint selection. The other 35 original files match the uploaded
ZIP byte-for-byte. The submitted figure order and omissions are preserved.
../revision_asset_changes.json records the submitted and revised hashes;
../reviewer_changes.patch covers the main source and text table. The archived
source ZIP, PDF, extracted original main/bibliography, and manifest stay unchanged.
The additional audit scripts, CSV/JSON evidence, and response/verification notes
document the corrections using the existing experiments.

To inspect the manuscript changes from the submitted baseline, run from the
repository root:

  diff -u docs/scalable_vits_for_embedded_systems/submission/bookchapter.tex docs/scalable_vits_for_embedded_systems/bookchapter.tex

../reviewer_changes.patch records that comparison for this revision. Regenerate
it after further manuscript edits; the source snapshots and archive stay fixed.
See ../revision_verification.txt for the version comparison and build evidence.

# Scorer provenance

`md-eval-22.pl` is the official NIST md-eval speaker-diarization scorer,
vendored from `nryant/dscore` at commit `e02f949` (SHA256 pinned in
`EVIDENCE.json`). Its `LICENSE` (BSD-2-Clause, dscore authors) is alongside.
No license change is claimed for the NIST tool itself.

Reference RTTM/UEM inputs embedded in `EVIDENCE.json` derive from
`BUTSpeechFIT/AMI-diarization-setup` at commit `2509d893` (`only_words`
refs, train split, n=4, not held out). BUT attribution from the original
reference README is preserved by this note.

# Model files acquired when a tool needs them

I added first-use acquisition to the native research jobs, TruFor, supported
Copy-Move learned matchers and Noiseprint. Before inference starts, the selected
method resolves its model dependencies, reuses verified local files and downloads
only missing resources. The progress display supports cancellation and resuming
partial transfers. No models are loaded into RAM by the download manager.

The versioned catalogue records 185 files across 78 method/dependency configurations,
with decoded sizes and SHA256. Existing GitHub RC1 parts support byte-range requests:
the downloader fetches only the compressed records it needs, including records that
cross part boundaries. D2PRL uses its separate GitHub asset. Downloads are streamed,
verified and installed atomically without overwriting unknown existing files.

Cancelling invalidates the pending UI result immediately. A network read can take up
to its 15-second timeout to stop. Partial bytes remain for a later attempt; corrupt
payloads are discarded. Shared dependencies are downloaded once, with an interprocess
lock. Inference uses each tool's existing loader only after its files are verified.

The optional offline preparation command verifies/downloads all catalogue resources:

```sh
python3 macos/install_models.py "/path/to/SHERLOQ-installation" --all
```

This command follows the [cumulative update](../README.md#apply-the-cumulative-rc1-updates).
The pre-existing optional missing-checkpoint list is separate: this catalogue does
not promise unavailable models. Tesseract and English OCR data are bundled with the
source update; they do not require a model download.

Transfer tests cover resume, corruption, range validation, local reuse, simultaneous
requests, cancellation and unknown-file protection. A real GitHub range download
and a first-use Adaptive CFA CPU inference were also tested in an independently
restored installation. [Integration results](SIFT-MODELS-PUBLIC-CHECKS.json).

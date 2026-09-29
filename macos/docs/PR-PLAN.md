# Contribution reference

I have retained SHERLOQ’s public upstream history and the original licenses. This fork contains the native application changes and installation support; its Release contains the frozen complete macOS environment.

| Area available for review | Reference |
| --- | --- |
| Implemented and restored tools | [Tool inventory](TOOLS.md) and [changelog](CHANGELOG.md) |
| Correctness fixes | Application helpers in `gui/sherloq_app/core/` and their calling widgets |
| Responsiveness and computation reuse | Caches, job lifecycle and worker services under `gui/sherloq_app/` |
| Recorded performance changes | [Measurements and scope](PERFORMANCE.md) |
| Native builds and restoration | [macOS guide](../README.md), `macos/packaging/` and vendor sources |
| Original project and component credits | [Attribution](ATTRIBUTION.md) and the retained license files |

The application changes share computation helpers and worker services. Copying individual modified files may omit required dependencies. The [file-level index](NATIVE-CHANGE-INDEX.md) maps the frozen contribution, and the original source-kit archive contains its transfer patch and upstream preimages. The complete candidate has not been presented as a set of independently validated patches.

# Proposed upstream contribution batches

The author may take any part of the archive without a public fork. The included
native.patch is a verified transfer patch; the following are review proposals,
not independently verified patch series yet.

1. Correctness: linear ELA signed-error loss, endpoint handling, decode ownership
   and quality-index fixes, with minimal synthetic regressions. Extract the
   required helpers with each fix; adapt to upstream layout before submission.
2. Shared performance architecture: cache budgets, retained computations, job
   lifecycle, cancellation and worker services. This is a dependency for several
   later tools, so do not split its consumers into broken intermediate revisions.
3. Tool families: JPEG/ELA, noise, comparison, copy-move and extra analysis as
   coherent families, each with parameter documentation and redistributable tests.
4. Native acceleration: build sources, CPU fallback, Metal paths and measured
   numerical comparisons. Keep optional kernels and their license boundaries clear.
5. Desktop interface: profiles, localization, gestures and region/display controls,
   together with their required computation/display contracts.
6. Browser interface and engines: a later, separately tested delivery from the web
   work; no placeholder implementation or demo claim belongs in these PRs.

Before each real PR: choose the contributor identity with the user, check the
current upstream base, extract and test the batch in order, and agree on author
preference (archive, targeted fork PRs, or an authorized branch). Nothing has been
pushed, sent or committed under an invented identity. The local repository uses
an unborn main branch with staged reviewed files and no remote.

# Bundled Hermes runtime

This directory contains the modified Hermes Agent 0.17.0 runtime used by the
Multi-Agent MPCS v5 workflow. Follow the repository [README](../README.md) for
installation, model/search configuration, runtime checks, and execution.

The source is based on [NousResearch/Hermes-Agent](https://github.com/NousResearch/Hermes-Agent),
base commit `1456f09e46bd842e7958e29c315681cddfe276f0`, with local changes to
auxiliary models, response transport, web retrieval, BrowseComp guards, and
request trace capture. This is not an unmodified upstream release.

The publication copy retains runtime code, dependency manifests and lockfiles,
plugin manifests, bundled skills and their reference resources, model catalog,
configuration examples, and component licenses. Development reports, plans,
tests, historical evaluation tools, documentation-site content, contributor
contact mappings, and release-maintenance files are excluded. Personal example
paths use generic usernames. The model catalog at
`website/static/api/model-catalog.json` remains because runtime code reads it.

Skill instructions and references are runtime resources; do not remove them
solely because they are Markdown files. Optional tools, gateways, and user
interfaces may require extra dependencies or their own build steps. The
repository README describes the supported Python workflow entrypoint.

Hermes is distributed under the [MIT license](LICENSE). Original copyright and
component license notices are retained. This license does not automatically
cover the separate workflow code in the parent directory.

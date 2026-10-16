# Releasing (maintainers)

Nothing in this repository uploads anything by itself; a release happens only when the
maintainer pushes a tag or runs `twine` with their own credentials.

1. Update `version` in `pyproject.toml` and `__version__` in `mab_benchmark/__init__.py`,
   and add a changelog entry.
2. Check locally: `pip install -e ".[dev]"`, `python -m pytest -q`,
   `python -m build`, `twine check dist/*`.
3. Either
   - **trusted publishing** (recommended): register the repository once at
     <https://pypi.org/manage/account/publishing/> (project `mab-benchmark`, workflow
     `release.yml`, environment `pypi`), then `git tag v1.3.0 && git push origin v1.3.0`;
     the `Release to PyPI` workflow builds and uploads; or
   - **by hand**: `twine upload dist/*` with a PyPI API token
     (username `__token__`).
4. Documentation: `mkdocs gh-deploy` publishes `docs/` to GitHub Pages.

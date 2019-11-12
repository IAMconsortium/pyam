
# Release procedure

## Required accounts and admin privileges

- pip: https://pypi.org/project/pyam-iamc/ and https://test.pypi.org/project/pyam-iamc/
- conda: https://github.com/conda-forge/pyam-feedstock/
- rtd: https://readthedocs.org/projects/pyam-iamc/

## Steps for publishing a new release

1. Make a release candidate branch (e.g., `rc_v<release version>`)
   and pull request it into `master` with the following updates:
   1. Deprecate any stated portion of the API
      (you can find them by searching the code base for "deprecate")
   1. Update `RELEASE_NOTES.md` (see the examples of previous releases)
	  - replace "# Next Release" with "# Release v<release version>"
	  - for this new release, add a "## Highlights" section with the most important updates & changes
      - add a new heading "## Individual Updates" before the PR listing
  1. Confirm that the PR passes all tests and checks
  1. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
     - **THIS IS NOT THE TAGGED COMMIT WE WILL DISTRIBUTE, IT IS ONLY FOR TESTING**
	 - **DO NOT PUSH THIS TAG TO UPSTREAM**
  1. Run `make publish-on-testpypi`
     - this should "just work" - if it does not, fix any issues,
       retag (`git tag -d` then `git tag`), and try again
  1. Once successful, delete the tag, and merge the candidate PR into `master` on Github
1. Switch to the now-updated master branch: `git checkout master` and `git pull upstream master`
1. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
   - `versioneer` automatically updates the version number based on the tag
   - this is now the official tagged commit
1. Push the tag upstream: `git push upstream --tags`
1. Run `make publish-on-pypi`
   - this will make wheels that allow `pyam` to be installed via `pip install`
   - check that the new version is available at https://pypi.org/project/pyam-iamc/
1. Make a new release on Github
   - make sure that you choose the tag name defined above
   - copy the release summary from `RELEASE_NOTES.md` into the description box
1. Update on `conda-forge`
   - a PR should automatically be opened by the bot after the Github release
   - confirm that any new depedencies are added there
   - merge the PR
   - check that the new version is available at https://anaconda.org/conda-forge/pyam
1. Confirm that the doc pages are updated on https://pyam-iamc.readthedocs.io/
   - both the **latest** and the **stable** versions point to the new release
   - the new release has been added to the list of available versions
1. Add a new line "# Next Release" at the top of `RELEASE_NOTES.md` and commit to `master`
1. Announce it on our mailing list: https://groups.io/g/pyam
   - again, copy the rendered HTML from the Github release directly in the email

And that's it! Whew...

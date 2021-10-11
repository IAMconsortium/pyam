# Release procedure

## Required accounts and admin privileges

- pip: https://pypi.org/project/pyam-iamc/ and https://test.pypi.org/project/pyam-iamc/
- conda: https://github.com/conda-forge/pyam-feedstock/
- rtd: https://readthedocs.org/projects/pyam-iamc/

## Steps for publishing a new release

1. Make a release candidate branch named `release/rc_v<release version>`
   and pull request it into `main` with the following updates:
   1. If it's the first release in a new year,
      search for `Copyright 2017` and bump the year
   1. Deprecate any portion of the API marked for deprecation in this release
      (you can find them by searching the code base for "deprecate")
   1. Update `RELEASE_NOTES.md` (see the examples of previous releases)
	  - replace "# Next Release" with "# Release v<release version>"
	  - add a "## Highlights" section with the most important updates & changes
      - if applicable, add/review "## Dependency changes" and "## API changes" sections 
      - add a new heading "## Individual Updates" before the PR listing
  1. Confirm that the PR passes all tests and checks
  1. Tag the release candidate: `git tag v<release version>rc<n>`,
     e.g., `git tag v1.2.0rc1`
  1. Confirm that the "test-publish" workflow passed 
     https://github.com/IAMconsortium/units/actions/workflows/test-publish.yml
  1. Confirm that the release is published on https://test.pypi.org/project/pyam-iamc/
     1. The package can be downloaded, installed and run
     1. The README is rendered correctly
  1. If there are any problems, fix the issues and repeat the step
     "tagging the release candidate", bumping the release candidate number  
  1. If successful, merge the candidate PR into `main` and then delete the branch
1. Switch to the now-updated main branch: `git checkout main` and `git pull upstream main`
1. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
   - `versioneer` automatically updates the version number based on the tag
   - this is now the official tagged commit
1. Push the tag upstream: `git push upstream --tags`
1. Make a new release on Github
   - make sure that you choose the tag name defined above
   - copy the release summary from `RELEASE_NOTES.md` into the description box
1. Confirm that the "publish" workflow passed 
   https://github.com/IAMconsortium/units/actions/workflows/publish.yml
1. Confirm that the release is published on https://www.pypi.org/project/pyam-iamc/
1. Update on `conda-forge`
   - a PR should automatically be opened by the bot after the Github release
   - confirm that any new depedencies are included,
     change the minimum dependency version if necessary
     (compare to ./.github/workflows/pytest-depedency.yml)
   - merge the PR
   - check that the new version is available at https://anaconda.org/conda-forge/pyam
1. Confirm that the doc pages are updated on https://pyam-iamc.readthedocs.io/
   - both the **latest** and the **stable** versions point to the new release
   - the new release has been added to the list of available versions
1. Add a new line "# Next Release" at the top of `RELEASE_NOTES.md` and commit to `main`
1. Announce it on our mailing list https://pyam.groups.io & social media (`#pyam_iamc`)
   - again, copy the rendered HTML from the Github release directly in the email

And that's it! Whew...

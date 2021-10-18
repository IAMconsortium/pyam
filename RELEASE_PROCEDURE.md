# Release procedure

## Required accounts and admin privileges

- pip: https://pypi.org/project/pyam-iamc/ and https://test.pypi.org/project/pyam-iamc/
- conda: https://github.com/conda-forge/pyam-feedstock/
- rtd: https://readthedocs.org/projects/pyam-iamc/

## Steps for publishing a new release

1. Create a release candidate branch named `release/rc_v<release version>`
   and pull request it into `main` with the following updates:
   1. If it's the first release in a new year,
      search for `Copyright 2017` and update the end-year of the copyright tag
   2. Deprecate any portion of the API marked for removal in this release
      (you can find them by searching the code base for "deprecate")
   3. Update `RELEASE_NOTES.md` (see the examples of previous releases)
	  - Replace "# Next Release" with "# Release v<release version>"
	  - Add a "## Highlights" section with the most important updates & changes
      - If applicable, add/review "## Dependency changes" and "## API changes" sections 
      - Add a new heading "## Individual Updates" before the list of individual PRs
   4. Confirm that the PR passes all tests
   5. Tag the release candidate `git tag v<release version>rc<n>`,
      e.g., `git tag v1.2.0rc1`, and push to the upstream repository
   6. Confirm that the "publish" workflow passes 
      https://github.com/IAMconsortium/pyam/actions/workflows/publish.yml
   7. Confirm that the release is published on https://test.pypi.org/project/pyam-iamc/
      - The package can be downloaded, installed and run
      - The README is rendered correctly
   8. If there are any problems, fix the issues and repeat from step 5,
      bumping the release candidate number  
   9. If successful, merge the candidate PR into `main` and then delete the branch
2. Switch to the updated main branch: `git checkout main` and `git pull upstream main`
3. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
4. Push the tag upstream: `git push upstream --tags`
5. Make a new release on Github
   - Make sure that you choose the tag name defined above
   - Copy the release summary from `RELEASE_NOTES.md` into the description box
6. Confirm that the "publish" workflow passes 
   https://github.com/IAMconsortium/pyam/actions/workflows/publish.yml
7. Confirm that the release is published on https://www.pypi.org/project/pyam-iamc/
8. Update on `conda-forge`
   - A PR should automatically be opened by the bot after the Github release
   - Confirm that any new depedencies are included,
     change the minimum dependency version if necessary
     (compare to ./.github/workflows/pytest-depedency.yml)
   - Merge the PR
   - Check that the new version is available on https://anaconda.org/conda-forge/pyam
9. Confirm that the doc pages are updated on https://pyam-iamc.readthedocs.io/
   - Both the **latest** and the **stable** versions point to the new release
   - The new release has been added to the list of available versions
10. Add a new first line "# Next Release" in `RELEASE_NOTES.md` and commit to `main`
11. Announce it to our community
    - The mailing list (https://pyam.groups.io) - copy the rendered HTML
      from the Github release and use the subject line `#release v<release version>`
    - The Slack channel
    - Social media using the tag `#pyam_iamc` 

And that's it! Whew...

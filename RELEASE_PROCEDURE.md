
# Release procedure

## Required accounts and admin privileges

- pip: https://pypi.org/project/pyam-iamc/ and https://test.pypi.org/project/pyam-iamc/
- conda: https://github.com/conda-forge/pyam-feedstock/
- rtd: https://readthedocs.org/projects/pyam-iamc/

## Steps for publishing a new release

1. Make a release candidate branch and pull request it into master with the following updates:
   1. Deprecate any stated portion of the API (you can find them by searching the code base for "deprecate")
   1. Update `RELEASE_NOTES.md` (see the examples of other releases)
	  - create new section under "# Next Release" with "# Release v<release version>"
	  - For this new release, add a "Highlights section" with some of the most important updates
      - Make a new heading "## Individual Updates" before the PR listing
  1. Confirm that the PR passes all tests and checks
  1. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
     - **THIS IS NOT THE TAGGED COMMIT WE WILL DISTRIBUTE, IT IS ONLY FOR TESTING**
	 - **DO NOT PUSH THIS TAG TO UPSTREAM**
  1. Run `make publish-on-testpypi`
     - this should "just work" if it does not, fix any issues, retag (`git tag
       -d` then `git tag`), and try again
	 - note, you need an account on https://test.pypi.org
  1. Once successful, delete the tag, and merge the candidate PR into master on Github
1. Switch to now updated master branch: `git checkout master` and `git pull upstream master`
1. Tag the release number: `git tag v<release version>`, e.g., `git tag v1.2.0`
   - `versioneer` automatically updates the version number based on the tag
   - this is now the official tagged commit
1. Push the tag upstream: `git push upstream --tags`
1. Run `make publish-on-pypi`
   - note, you need an account on https://pypi.org
   - this will make wheels that all us to be installed via `pip install`
1. Make a new release on Github
   - Make sure that you choose the same tag name as was used earlier
   - Copy the markdown from `RELEASE_NOTES.md` into the release description box
1. Update on conda-forge:
   - A PR should automatically be opened by the bot after the Github release
   - confirm that any new depedencies are added there
1. Announce it on our mailing list: https://groups.io/g/pyam
   - Again, just copy the now rendered HTML from the Github release directly in
     the email
1. Confirm that the doc page is updated to the latest release: https://pyam-iamc.readthedocs.io/

And that's it! Whew...

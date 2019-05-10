
# Steps required for a release

#. Make a release candidate branch and pull request it into master with the following updates:
   #. Deprecate any stated portion of the API (you can find them by searching the code base for "deprecate")
   #. Update `RELEASE_NOTES.md` (see the examples of other releases)
	  - create new section under "# Next Release" with "# Release v<release version>"
	  - For this new release, add a "Highlights section" with some of the most important updates
      - Make a new heading "## Individual Updates" before the PR listing
#. After tests pass, merge the PR
#. Switch to now updated master branch: `git checkout master` and `git pull upstream master`
#. Tag the release number: `git tag v<release version>`, e.g., `git tag v0.2.0`
   - `versioneer` automatically updates the version number based on the tag
#. 


# Steps required for a release

#. Make a release candidate branch and pull request it into master with the following updates:
   #. Deprecate any stated portion of the API (you can find them by searching the code base for "deprecate")
   #. Update `RELEASE_NOTES.md` (see the examples of other releases)
	  - create new section under "# Next Release" with "# Release v<release version>"
	  - For this new release, add a "Highlights section" with some of the most important updates
      - Make a new heading "## Individual Updates" before the PR listing
  #. Confirm that the PR passes all tests and checks
  #. Tag the release number: `git tag v<release version>`, e.g., `git tag v0.2.0`
     - **THIS IS NOT THE TAGGED COMMIT WE WILL DISTRIBUTE, IT IS ONLY FOR TESTING**
	 - **DO NOT PUSH THIS TAG TO UPSTREAM**
  #. Run `make publish-on-testpypi`
     - this should "just work" if it does not, fix any issues, retag (`git tag
       -d` then `git tag`), and try again
	 - note, you need an account on https://test.pypi.org
  #. Once successful, delete the tag, and merge the candidate PR into master on Github
#. Switch to now updated master branch: `git checkout master` and `git pull upstream master`
#. Tag the release number: `git tag v<release version>`, e.g., `git tag v0.2.0`
   - `versioneer` automatically updates the version number based on the tag
   - this is now the official tagged commit
#. Push the tag upstream: `git push upstream --tags`
#. Run `make publish-on-pypi`
   - note, you need an account on https://pypi.org
   - this will make wheels that all us to be installed via `pip install`
#. Update on conda-forge:
   - Issue a PR on https://github.com/conda-forge/pyam-feedstock following the
     instructions there for how to edit the `recipe/meta.yaml` file
   - Note that you can get the correct SHA256 hash from
     https://pypi.org/project/pyam-iamc/#files once that step has been
     successful
#. Make a new release on Github
   - Make sure that you choose the same tag name as was used earlier
   - Copy the markdown from `RELEASE_NOTES.md` into the release description box
#. Announce it on our mailing list: https://groups.google.com/forum/#!forum/pyam
   - Again, just copy the now rendered HTML from the Github release directly in
     the email
#. Confirm that the doc page is updated to the latest release: https://pyam-iamc.readthedocs.io/

And that's it! Whew...

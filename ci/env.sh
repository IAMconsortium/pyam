
# setup miniconda url
case "${TRAVIS_OS_NAME}" in
    linux)
        OSNAME=Linux
        EXT=sh
    ;;
    osx)
        OSNAME=MacOSX
        EXT=sh
    ;;
    windows)
        OSNAME=Windows
        EXT=exe
    ;;
esac

case "${PYENV}" in
    py27)
        PYVERSION=2
	CHOCONAME='miniconda'
    ;;
    py37)
        PYVERSION=3
	CHOCONAME='miniconda3'
    ;;
esac

if [[ "$TRAVIS_OS_NAME" != 'windows' ]]; then
    export PATH=$HOME/miniconda/bin:$PATH
else
    export PATH="/c/tools/$CHOCONAME/scripts:/c/tools/$CHOCONAME/:$PATH"
fi

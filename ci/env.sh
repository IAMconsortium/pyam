
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
        export PYVERSION=2
        export CHOCOPATH='miniconda2'
        export CHOCONAME='miniconda'
    ;;
    py37)
        export PYVERSION=3
        export CHOCOPATH='miniconda3'
        export CHOCONAME='miniconda3'
    ;;
esac

if [[ "$TRAVIS_OS_NAME" != 'windows' ]]; then
    export PATH=$HOME/miniconda/bin:$PATH
else
    export PATH="/c/tools/$CHOCOPATH/scripts:/c/tools/$CHOCOPATH/:$PATH"
fi

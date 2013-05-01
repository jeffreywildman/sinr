#!/bin/sh -e

test -n "$srcdir" || srcdir=`dirname "$0"`
test -n "$srcdir" || srcdir=.
(
  cd "$srcdir" &&
  autoreconf --warnings=all --force --install
  # Patch ltmain.sh so that --as-needed works also when creating shared libs
  # (as libcontentaction does it)
  # Patch it BEFORE ./configure is run!
  if patch -s -t -p0 --dry-run $srcdir/config/ltmain.sh $srcdir/config/libtool-2.4.2-as-needed.patch; then
  	patch -t -p0 $srcdir/config/ltmain.sh $srcdir/config/libtool-2.4.2-as-needed.patch
  else
    echo "WARNING: libtool-2.4.2-as-needed.patch not applied."
  fi
) || exit
test -n "$NOCONFIGURE" || "$srcdir/configure" "$@"

exit

#!/usr/bin/env bash
set -e

# === Configuration ===
SRC_DIR="src/porespy/beta"
TARGET_DIR="$SRC_DIR/build"

echo "Setting up build directory..."
uv run meson setup "$TARGET_DIR" "$SRC_DIR" --buildtype=plain --vsenv --reconfigure

echo "Compiling C extension..."
uv run meson compile -C "$TARGET_DIR"

echo "Moving compiled library into Python package..."
case "$OSTYPE" in
linux*)
  mv "$TARGET_DIR/libwalker.so" "$SRC_DIR/"
  echo "Moved libwalker.so"
  ;;
darwin*)
  mv "$TARGET_DIR/libwalker.dylib" "$SRC_DIR/"
  echo "Moved libwalker.dylib"
  ;;
*)
  echo "Unsupported OS or shell: $OSTYPE"
  exit 1
  ;;
esac

echo "Build complete."

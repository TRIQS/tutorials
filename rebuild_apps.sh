#!/usr/bin/env bash
# Refresh ~/opt/triqs with the latest TRIQS + tutorial-relevant applications.
#
# Usage:
#   ./rebuild_apps.sh             # build + install everything (no git pull)
#   ./rebuild_apps.sh --pull      # fast-forward each repo on its current branch first
#   ./rebuild_apps.sh --apps a,b  # build only the listed apps (still installs TRIQS first)
#
# Assumes each $SRC/$app has a configured build/ directory pointing at
# CMAKE_INSTALL_PREFIX=~/opt/triqs. Re-runs `cmake --build` + `cmake --install`.

set -euo pipefail

SRC="${SRC:-$HOME/Dropbox/Coding}"
PREFIX="$HOME/opt/triqs"
JOBS="${JOBS:-8}"

PULL=0
APPS_FILTER=""
while (( $# )); do
  case "$1" in
    --pull)          PULL=1; shift;;
    --apps)          APPS_FILTER="$2"; shift 2;;
    --jobs|-j)       JOBS="$2"; shift 2;;
    -h|--help)       sed -n '1,15p' "$0"; exit 0;;
    *)               echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# TRIQS first (from triqs_unstable — main triqs/ checkout may be on a feature branch).
# Then apps in dependency order. cppdlr2d is a math lib modest links to at runtime.
declare -a STAGES=(
  "triqs:triqs_unstable"
  "cthyb:cthyb"
  "ctseg:ctseg"
  "maxent:maxent"
  "hubbardI:hubbardI"
  "dft_tools:dft_tools"
  "dftkit:dftkit"
  "cppdlr2d:cppdlr2d"
  "modest:modest"
  "solid_dmft:solid_dmft"
)

# Apps that need to be built as shared libraries when installed into $PREFIX.
declare -A SHARED_LIBS=(
  [cppdlr2d]=1
)

filter_ok() {
  [[ -z "$APPS_FILTER" ]] && return 0
  local app="$1"
  IFS=',' read -ra wanted <<<"$APPS_FILTER"
  for w in "${wanted[@]}"; do [[ "$w" == "$app" ]] && return 0; done
  return 1
}

log() { printf '\n\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN: %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }
fail() { printf '\033[1;31m[%s] FAIL: %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }

pull_ff() {
  local dir="$1"
  cd "$dir"
  local branch
  branch=$(git rev-parse --abbrev-ref HEAD)
  if [[ "$branch" == "HEAD" ]]; then
    warn "$dir is in detached HEAD — skipping pull"
    return 0
  fi
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    warn "$dir has uncommitted changes — skipping pull"
    return 0
  fi
  log "git pull --ff-only in $(basename "$dir") ($branch)"
  git fetch --quiet || { warn "fetch failed in $dir"; return 0; }
  git pull --ff-only --quiet || { warn "pull --ff-only failed in $dir (non-ff?)"; return 0; }
}

build_install() {
  local app="$1" dir="$2"
  local d="$SRC/$dir"
  local shared_flag=""
  [[ "${SHARED_LIBS[$app]:-0}" == "1" ]] && shared_flag="-DBUILD_SHARED_LIBS=ON"

  # If the build dir doesn't exist OR was configured against a different
  # install prefix OR resolved TRIQS / cppdlr2d from a different prefix,
  # do a full reset to drop stale *_DIR cache entries.
  local needs_reset=0
  if [[ ! -f "$d/build/CMakeCache.txt" ]]; then
    needs_reset=1
  else
    local cur
    cur=$(grep -E '^CMAKE_INSTALL_PREFIX:PATH=' "$d/build/CMakeCache.txt" | cut -d= -f2- || true)
    [[ "$cur" != "$PREFIX" ]] && needs_reset=1
    # Drift checks: TRIQS-stack *_DIR entries that resolved outside $PREFIX mean stale
    # find_package results from a different install (e.g. triqs_dlr2d).
    local pkg
    for pkg in TRIQS cppdlr cppdlr2d c2py nda h5 mpi itertools Cpp2Py; do
      local found
      found=$(grep -E "^${pkg}_DIR:PATH=" "$d/build/CMakeCache.txt" | cut -d= -f2- || true)
      if [[ -n "$found" && "$found" != "$PREFIX"/* ]]; then
        log "[$app] stale ${pkg}_DIR=$found (want under $PREFIX) — will reset"
        needs_reset=1
        break
      fi
    done
  fi

  if (( needs_reset )); then
    log "[$app] wiping build/ (stale or missing CMake cache)"
    rm -rf "$d/build" && mkdir -p "$d/build"
    log "[$app] configure (prefix=$PREFIX${shared_flag:+, $shared_flag})"
    cmake -S "$d" -B "$d/build" -GNinja \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      $shared_flag
  fi

  log "[$app] build"
  cmake --build "$d/build" -j "$JOBS"
  log "[$app] install"
  cmake --install "$d/build" >/dev/null
}

# 0. sanity
if [[ ! -d "$SRC" ]]; then
  echo "SRC=$SRC does not exist" >&2; exit 1
fi

# 1. optional pulls
if (( PULL )); then
  for stage in "${STAGES[@]}"; do
    app="${stage%%:*}"; dir="${stage##*:}"
    filter_ok "$app" || continue
    [[ -d "$SRC/$dir" ]] || { warn "$SRC/$dir missing — skip"; continue; }
    pull_ff "$SRC/$dir"
  done
fi

# 2. always build TRIQS first
log "Refreshing TRIQS install at $PREFIX"
build_install "triqs" "triqs_unstable"

# 3. source the freshly installed env so apps pick it up
# shellcheck source=/dev/null
source "$PREFIX/share/triqs/triqsvars.sh"

# 4. apps
for stage in "${STAGES[@]}"; do
  app="${stage%%:*}"; dir="${stage##*:}"
  [[ "$app" == "triqs" ]] && continue
  filter_ok "$app" || continue
  [[ -d "$SRC/$dir" ]] || { warn "$SRC/$dir missing — skip"; continue; }
  # Tolerant per-app: keep going on failure but record it.
  if ! build_install "$app" "$dir"; then
    fail "$app failed to build/install — continuing"
  fi
done

log "Done."
log "Installed Python packages in $PREFIX/lib/python*/site-packages:"
ls "$PREFIX"/lib/python*/site-packages/ 2>/dev/null

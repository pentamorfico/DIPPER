#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIPPER="${DIPPER_BIN:-${ROOT}/bin/dipper}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

if [[ ! -x "${DIPPER}" ]]; then
    echo "missing executable: ${DIPPER}" >&2
    exit 2
fi

# Includes identical, conservative, radical, gapped and partially-overlapping
# protein pairs. The first identifier exercises Newick quoting.
cat > "${TMP}/proteins.fa" <<'EOF'
>identical:sample
ACDEFGHIKLMNPQRS
>conservative sample
ACDEFGHIKLMNPQRT
>radical(sample)
WWWWWWWWWWWWWWWW
>gapped[partial]
ACDE----KLMN----
EOF

run_model() {
    local model="$1"
    "${DIPPER}" --protein --protein-model "${model}" --entropy-gamma 2.0 \
        --gamma-alpha 0.8 --protein-saturation 0.92 --overlap-prior 3.0 \
        -i m -o t -d 2 -I "${TMP}/proteins.fa" -O "${TMP}/${model}.nwk" \
        >"${TMP}/${model}.log" 2>&1
    test -s "${TMP}/${model}.nwk"
    ! grep -Eiq 'nan|inf' "${TMP}/${model}.nwk"
    grep -q "'identical:sample'" "${TMP}/${model}.nwk" || {
        echo "Newick did not quote reserved leaf name for ${model}:" >&2
        cat "${TMP}/${model}.nwk" >&2
        exit 1
    }
}

run_model blosum62
run_model binary

if cmp -s "${TMP}/blosum62.nwk" "${TMP}/binary.nwk"; then
    echo "binary and BLOSUM62 unexpectedly produced identical output" >&2
    exit 1
fi

echo "protein model smoke tests passed"

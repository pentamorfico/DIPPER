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
cp "${ROOT}/tests/data/proteins.fa" "${TMP}/proteins.fa"

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

# Force the divide-and-conquer path.  This specifically guards initialization
# of the constant-memory amino-acid matrix: when it is left at CUDA's zero
# initialization, every binary p-distance becomes 1 and this radical branch
# no longer has the expected high distance.
"${DIPPER}" --protein --protein-model binary --entropy-gamma 0 \
    --overlap-prior 0 -i m -o t -d 1 -m 3 --max-cluster-size 3 \
    -I "${ROOT}/tests/data/proteins_sanitized.fa" -O "${TMP}/dc-binary.nwk" \
    >"${TMP}/dc-binary.log" 2>&1
test -s "${TMP}/dc-binary.nwk"
! grep -Eiq 'nan|inf' "${TMP}/dc-binary.nwk"
grep -Eq 'radical_sample:0\.(8|9)[0-9]*' "${TMP}/dc-binary.nwk" || {
    echo "DC protein matrix appears uninitialized:" >&2
    cat "${TMP}/dc-binary.nwk" >&2
    exit 1
}

echo "protein model smoke tests passed"

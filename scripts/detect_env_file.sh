#!/usr/bin/env bash
set -euo pipefail

# Determine the directory to start searching from. If a path is provided use it,
# otherwise default to the directory containing this script (the project
# repository root).
START_DIR=${1:-""}
if [[ -z "${START_DIR}" ]]; then
    START_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    if [[ -d "${START_DIR}" ]]; then
        START_DIR="$(cd "${START_DIR}" && pwd)"
    else
        echo "Error: ${START_DIR} is not a directory." >&2
        exit 1
    fi
fi

find_env_file() {
    local dir="$1"
    while [[ -n "${dir}" && "${dir}" != "/" ]]; do
        if [[ -f "${dir}/.env" ]]; then
            printf '%s\n' "${dir}/.env"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    return 1
}

copy_template_to_env() {
    local dir="$1"
    local template
    for candidate in ".env.template" ".template.env"; do
        template="${dir}/${candidate}"
        if [[ -f "${template}" ]]; then
            local target="${dir}/.env"
            if [[ -f "${target}" ]]; then
                printf '%s\n' "${target}"
                return 0
            fi
            if cp "${template}" "${target}"; then
                echo "Created ${target} from ${candidate}. Update the values before running services." >&2
                printf '%s\n' "${target}"
                return 0
            else
                echo "Error: Unable to create ${target} from ${candidate}." >&2
                return 1
            fi
        fi
    done
    return 1
}

if env_path="$(find_env_file "${START_DIR}")"; then
    printf '%s\n' "${env_path}"
    exit 0
fi

# If no environment file was found walk up the tree looking for a template we
# can copy into place.
search_dir="${START_DIR}"
while [[ -n "${search_dir}" && "${search_dir}" != "/" ]]; do
    if env_path="$(copy_template_to_env "${search_dir}")"; then
        printf '%s\n' "${env_path}"
        exit 0
    fi
    search_dir="$(dirname "${search_dir}")"
done

echo "Warning: No .env file or template found starting from ${START_DIR}." >&2
exit 1

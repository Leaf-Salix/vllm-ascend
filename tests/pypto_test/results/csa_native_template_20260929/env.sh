#!/usr/bin/env bash
# Source after env-dsv4-0251rc1.sh for both Native and PTO template comparisons.
csa_template_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
csa_template_opp="$csa_template_repo/.cache/csa/native-template-cann92-opp/opp"
csa_template_vendor="$csa_template_repo/.cache/csa/native-template-cann92-stage/packages/vendors/csa_template_transformer"
if [[ ! -f "$csa_template_vendor/op_api/lib/libcust_opapi.so" || ! -d "$csa_template_opp" ]]; then
    echo "Build the Native template dependency and run prepare_opp.py first." >&2
    return 1
fi
export ASCEND_OPP_PATH="$csa_template_opp"
case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":$csa_template_vendor:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="$csa_template_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
esac
unset csa_template_repo csa_template_opp csa_template_vendor

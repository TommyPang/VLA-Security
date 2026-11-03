# Auxiliary MolmoAct2 diagnostics

This is a separate track from the main MolmoAct/TRAP experiment. It instruments
an existing MolmoAct2 LeRobot/LIBERO evaluation to inspect scene text, VLM
activations, the K/V interface, and action-expert responses.

Run `python3 scripts/bootstrap_sources.py --track molmoact2` from the repository
root to fetch the recorded wrapper and apply our modifications. The bootstrap
also copies these diagnostic modules beside the wrapper. Then follow
[INTERPRETABILITY.md](INTERPRETABILITY.md) in that initialized checkout.

This workspace does not contain the collaborator's full LeRobot environment,
fine-tuned policy, sign images, or `security_attack_utils.composite_sign` scene
helper. Supply those existing dependencies and explicit `SIGN_RGBA_PATH`,
`BENIGN_SIGN_RGBA_PATH`, and `NONSENSE_SIGN_RGBA_PATH` values before running the
visual conditions. The diagnostics code and wrapper patch are preserved here;
this auxiliary track is not a complete standalone reproduction of that
external setup. No MolmoAct2 outcome is claimed by the main experiment table.

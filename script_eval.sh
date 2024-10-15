#/usr/bin/env bash
#
# A Bash script used to run all the evaluations of the training experiments.
# The script assumes that all experiments are stored in the "results/final"
# directory and that the names match two patterns:
#
#   - "DFAAS-MA_{ENV NAME}_500_{real, synt...}" for a simple PPO experiment,
#
#   - DFAAS-MA_{ENV NAME}_500_cc_{real, synth...}" for experiments using PPO
#   with centralized critic.
shopt -s nullglob

for exp in results/final/DFAAS-MA_*_500_{r,s}*; do
  echo "++ Evaluating $exp"
  python dfaas_evaluate_ppo.py "$exp"  --seed 26 --env-config synt_sinusoidal_env_config.json synt_sin
  python dfaas_evaluate_ppo.py "$exp" --seed 26 --env-config synt_normal_env_config.json synt_norm
  python dfaas_evaluate_ppo.py "$exp" --seed 26 --env-config real_env_config.json real
done

for exp in results/final/DFAAS-MA_*_500_cc_*; do
  echo "++ Evaluating $exp"
  python dfaas_evaluate_centralized_ppo.py "$exp" --seed 26 --env-config synt_sinusoidal_env_config.json synt_sin
  python dfaas_evaluate_centralized_ppo.py "$exp" --seed 26 --env-config synt_normal_env_config.json synt_norm
  python dfaas_evaluate_centralized_ppo.py "$exp" --seed 26 --env-config real_env_config.json real
done

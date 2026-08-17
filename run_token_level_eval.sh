#!/usr/bin/env bash
# Token-level evaluation on the held-out test set, O excluded, for the
# MULTILABEL all_entities models.
#
# Decoding is the native multilabel one (sigmoid > 0.5), identical to what
# MultiLabelTrainer.compute_metrics uses during training. The only difference
# from the numbers already in fold_N_results.json is that O is dropped from the
# micro/macro aggregates.
#
# Results:
#   <experiment>/token_eval_holdout/         main run
#   <experiment>/token_eval_holdout_<lang>/  per-language runs (multilingual model)

set -u

source /home/z0048cmk/dt4h/bin/activate

BASE_DIR="/home/z0048cmk/DT4H/NER_task/CardioLM_paper_experiments/CardioLM"
export PYTHONPATH="${BASE_DIR}/CardioNER/src:${PYTHONPATH:-}"

DATA_DIR="${BASE_DIR}/assests/cardioccc_v1.3/b1+b2_without_sugs"
SPLITS_FILE="${BASE_DIR}/dataset_splits/splits_cv10_holdout_test.json"
MODEL_TOP="/mnt/DATA/DT4H/NER_task/CardioLM_paper_models/trained_NER_models"
EVAL_SCRIPT="${BASE_DIR}/run_token_level_holdout.py"

# Must match training (run_main.sh); a mismatch silently changes the token set.
CHUNK_SIZE=256
CHUNK_TYPE=paragraph
MAX_LENGTH=256

MODEL_TYPES=(
    multilabel_cardioberta
    multilabel_robeczech_base
    multilabel_xlm_roberta_base
    multilabel_xlm_roberta_large
)

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for EXPERIMENT_DIR in "${MODEL_TOP}/${MODEL_TYPE}"/*/; do
        EXPERIMENT="$(basename "$EXPERIMENT_DIR")"

        # Skip result/aux dirs, keep only real experiments with fold models
        [[ "$EXPERIMENT" == inference_eval* || "$EXPERIMENT" == token_eval* ]] && continue
        [[ "$EXPERIMENT" == merged_model || "$EXPERIMENT" == ablation ]] && continue
        [[ -d "${EXPERIMENT_DIR}fold_0" ]] || continue

        # cz_all_entities -> lang=cz, entity=all_entities
        LANG="${EXPERIMENT%%_*}"
        ENTITY="${EXPERIMENT#*_}"

        # Czech: only the lr_7e-5 runs. Other languages: the plain all_entities run.
        if [[ "$LANG" == "cz" ]]; then
            [[ "$ENTITY" == "all_entities_lr_7e-5" ]] || continue
        else
            [[ "$ENTITY" == "all_entities" ]] || continue
        fi

        if [[ "$LANG" == "multilingual" ]]; then
            BULK_FILE="${DATA_DIR}/annotations_multilingual_all_entities.jsonl"
        else
            BULK_FILE="${DATA_DIR}/${LANG}/annotations_all_entities.jsonl"
        fi

        OUTPUT_DIR="${EXPERIMENT_DIR}token_eval_holdout"

        [[ -f "${OUTPUT_DIR}/aggregated_holdout_token_results.json" ]] && continue
        [[ -f "$BULK_FILE" ]] || { echo "MISSING BULK: $BULK_FILE"; continue; }

        echo ""
        echo "=== $MODEL_TYPE/$EXPERIMENT (token-level, O excluded) ==="

        if ! python "$EVAL_SCRIPT" \
            --bulk_file "$BULK_FILE" \
            --split_file "$SPLITS_FILE" \
            --model_folder "$EXPERIMENT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --chunk_size "$CHUNK_SIZE" \
            --chunk_type "$CHUNK_TYPE" \
            --max_length "$MAX_LENGTH" \
            --fp16
        then
            echo "FAILED: $MODEL_TYPE/$EXPERIMENT -- stopping."
            exit 1
        fi
    done
done

# ---- per-language evaluation of the multilingual model ---------------------
# Same 51 held-out document ids, scored against each language's own corpus.

MULTILINGUAL_DIR="${MODEL_TOP}/multilabel_xlm_roberta_large/multilingual_all_entities/"

if [[ -d "${MULTILINGUAL_DIR}fold_0" ]]; then
    for LANG in cz en es it nl ro sv; do
        BULK_FILE="${DATA_DIR}/${LANG}/annotations_all_entities.jsonl"
        OUTPUT_DIR="${MULTILINGUAL_DIR}token_eval_holdout_${LANG}"

        [[ -f "${OUTPUT_DIR}/aggregated_holdout_token_results.json" ]] && continue
        [[ -f "$BULK_FILE" ]] || { echo "MISSING BULK: $BULK_FILE"; continue; }

        echo ""
        echo "=== multilingual_all_entities (token-level, per-lang=$LANG) ==="

        if ! python "$EVAL_SCRIPT" \
            --bulk_file "$BULK_FILE" \
            --split_file "$SPLITS_FILE" \
            --model_folder "$MULTILINGUAL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --chunk_size "$CHUNK_SIZE" \
            --chunk_type "$CHUNK_TYPE" \
            --max_length "$MAX_LENGTH" \
            --fp16
        then
            echo "FAILED: multilingual per-lang=$LANG -- stopping."
            exit 1
        fi
    done
fi

echo ""
echo "DONE (token-level holdout evaluation, O excluded)"

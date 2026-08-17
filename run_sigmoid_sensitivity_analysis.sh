#!/usr/bin/env bash
# Span-level decoding ablation for the MULTILABEL models.
#
# Re-runs fold inference with the confidence scale used by the decoding
# thresholds swapped from softmax to sigmoid. Label selection is unchanged
# (argmax over raw logits), so any difference comes from thresholding alone.
#
# Set ACTIVATION below:
#   sigmoid -> the ablation
#   softmax -> control run; should reproduce the existing inference_eval numbers
#
# All output is confined to <experiment>/ablation/, so the reported
# inference_eval/ results are never touched:
#   <experiment>/ablation/<activation>/          main run
#   <experiment>/ablation/<activation>_<lang>/   per-language multilingual runs

set -u

ACTIVATION="sigmoid"  # "softmax" or "sigmoid"

source /home/z0048cmk/dt4h/bin/activate

BASE_DIR="/home/z0048cmk/DT4H/NER_task/CardioLM_paper_experiments/CardioLM"
export PYTHONPATH="${BASE_DIR}/CardioNER/src:${PYTHONPATH:-}"

DATA_DIR="${BASE_DIR}/assests/cardioccc_v1.3/b1+b2_without_sugs"
SPLITS_FILE="${BASE_DIR}/dataset_splits/splits_cv10_holdout_test.json"
MODEL_TOP="/mnt/DATA/DT4H/NER_task/CardioLM_paper_models/trained_NER_models"

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
        [[ "$EXPERIMENT" == inference_eval* ]] && continue
        [[ "$EXPERIMENT" == merged_model || "$EXPERIMENT" == token_eval ]] && continue
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

        # The lr suffix names the model dir only; the data is the same either way.
        ENTITY="all_entities"

        if [[ "$LANG" == "multilingual" ]]; then
            BULK_FILE="${DATA_DIR}/annotations_multilingual_${ENTITY}.jsonl"
            LANG="multi"
        else
            BULK_FILE="${DATA_DIR}/${LANG}/annotations_${ENTITY}.jsonl"
        fi

        OUTPUT_DIR="${EXPERIMENT_DIR}ablation/${ACTIVATION}"

        [[ -f "${OUTPUT_DIR}/aggregated_validation_results.json" ]] && continue
        [[ -f "$BULK_FILE" ]] || { echo "MISSING BULK: $BULK_FILE"; continue; }

        echo ""
        echo "=== $MODEL_TYPE/$EXPERIMENT  (lang=$LANG, act=$ACTIVATION) ==="

        if ! python "${BASE_DIR}/run_sigmoid_ablation.py" \
            --activation "$ACTIVATION" \
            --bulk_file "$BULK_FILE" \
            --split_file "$SPLITS_FILE" \
            --model_folder "$EXPERIMENT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --lang "$LANG" \
            --pipe dt4h
        then
            echo "FAILED: $MODEL_TYPE/$EXPERIMENT -- stopping."
            exit 1
        fi
    done
done

# ---- per-language evaluation of the multilingual model ---------------------
# The multilingual model is also evaluated on each language's test set
# separately. Output goes to inference_eval_<lang>_<activation>/, alongside the
# existing inference_eval_<lang>/ results.

MULTILINGUAL_DIR="${MODEL_TOP}/multilabel_xlm_roberta_large/multilingual_all_entities/"

if [[ -d "${MULTILINGUAL_DIR}fold_0" ]]; then
    for LANG in cz en es it nl ro sv; do
        BULK_FILE="${DATA_DIR}/${LANG}/annotations_all_entities.jsonl"
        OUTPUT_DIR="${MULTILINGUAL_DIR}ablation/${ACTIVATION}_${LANG}"

        [[ -f "${OUTPUT_DIR}/aggregated_validation_results.json" ]] && continue
        [[ -f "$BULK_FILE" ]] || { echo "MISSING BULK: $BULK_FILE"; continue; }

        echo ""
        echo "=== multilingual_all_entities  (per-lang=$LANG, act=$ACTIVATION) ==="

        if ! python "${BASE_DIR}/run_sigmoid_ablation.py" \
            --activation "$ACTIVATION" \
            --bulk_file "$BULK_FILE" \
            --split_file "$SPLITS_FILE" \
            --model_folder "$MULTILINGUAL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --lang "$LANG" \
            --pipe dt4h
        then
            echo "FAILED: multilingual per-lang=$LANG -- stopping."
            exit 1
        fi
    done
fi

echo ""
echo "DONE (activation=$ACTIVATION)"
